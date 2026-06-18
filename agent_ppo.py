import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim import AdamW
from transformers import BertModel, BertConfig, PreTrainedTokenizerFast
from utils import Regression
import numpy as np
import re


class TransformerStateEncoder(nn.Module):
    """
    A dedicated module to encode textual state sequences into embeddings using a Transformer.
    Handles tokenization, numerical value injection, and BERT processing.
    """
    def __init__(self, input_bit, max_p, max_len, vocab_path='./vocab.json'):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=vocab_path)
        self.tokenizer.pad_token = "[PAD]"

        new_tokens = set([
            "CLS", "SEP", "IDLE", "NUM_PLACEHOLDER", "COL", "DOTS", "SIG", "dp", "NEXT",
            "G_PWR", "G_DLY", "G_DLY_CONSTRAINT", "NXT_STG", "TERMINAL_STATE", 'ACTIVE_STAGE', 'STEP', 'LIMIT'
        ])

        for i in range(input_bit):
            for j in range(input_bit):
                new_tokens.add(f"ip_{i}_{j}")
        for i in range(max_p + 1):
            new_tokens.add(f"pin{i}")
        for i in range(int(0.5*(input_bit**2+input_bit+5))):
            new_tokens.add(f"{i}")

        self.tokenizer.add_tokens(list(new_tokens))
        vocab_size = len(self.tokenizer)

        config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=128,
            num_hidden_layers=3,
            num_attention_heads=4,
            intermediate_size=512,
            max_position_embeddings=1600,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1
        )
        self.transformer = BertModel(config)
        self.hidden_size = config.hidden_size

        self.value_embedder = nn.Sequential(
            nn.Linear(1, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, config.hidden_size)
        )

    def _extract_numerical_values(self, text):
        """Extracts numerical values from text and replaces them with a placeholder."""
        numerical_values = []
        if not isinstance(text, str):
            text = str(text)

        def replace_float(match):
            value = float(match.group())
            numerical_values.append(value)
            return "NUM_PLACEHOLDER"

        text = re.sub(r'\b\d+\.\d+\b', replace_float, text)
        return text, numerical_values

    def _process_and_embed(self, text_batch, max_len):
        """Tokenizes, embeds, and injects numerical features for a batch of text sequences."""
        processed_texts, all_numerals = [], []
        for text in text_batch:
            processed, numerals = self._extract_numerical_values(text)
            processed_texts.append(processed)
            all_numerals.append(numerals)

        pre_tokenized = [s.split() for s in processed_texts]
        tokens = self.tokenizer(pre_tokenized, is_split_into_words=True, return_tensors='pt',
                                padding='max_length', truncation=True, max_length=max_len).to(self.device)

        last_hidden_state = self.transformer(**tokens).last_hidden_state

        placeholder_id = self.tokenizer.convert_tokens_to_ids("NUM_PLACEHOLDER")
        num_positions_mask = (tokens['input_ids'] == placeholder_id)

        for i in range(last_hidden_state.shape[0]):
            numerals = all_numerals[i]
            positions = num_positions_mask[i].nonzero().squeeze(-1)
            if numerals and len(numerals) == len(positions):
                values_tensor = torch.tensor(numerals, dtype=torch.float32, device=self.device).unsqueeze(-1)
                num_embeddings = self.value_embedder(values_tensor)
                last_hidden_state[i, positions] += num_embeddings

        return last_hidden_state

    def forward(self, text_batch, max_len):
        """Processes a batch of text sequences and returns their [CLS] embeddings."""
        embeddings = self._process_and_embed(text_batch, max_len)
        return embeddings


class BetterTransformerPolicy(nn.Module):
    """
    An Actor-Critic policy using a Transformer backbone and Cross-Attention for state-action fusion.
    """

    def __init__(self, input_bit, max_p, max_len, vocab_path='./vocab.json', lr=1e-4):
        super().__init__()
        self.lr = lr
        self.max_len = max_len
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.state_encoder = TransformerStateEncoder(input_bit, max_p, max_len, vocab_path)
        self.action_encoder = TransformerStateEncoder(input_bit, max_p, 16, vocab_path) # Actions have shorter max_len
        
        config = self.state_encoder.transformer.config

        # --- Critic Head ---
        # Predicts the value of a state (V(s))
        self.value_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, 1)
        )

        # --- Actor Head with Cross-Attention ---
        # Fuses state and action information to produce action scores
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            batch_first=True
        )
        self.actor_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, 1)
        )

    def forward(self, state_sequence_batch, action_sequences_batch):
        """
        Processes a batch of states and their corresponding action sets.
        Returns action scores for the Actor and state values for the Critic.
        """
        # --- 1. Process States (for both Actor and Critic) ---
        state_embeddings = self.state_encoder(state_sequence_batch, self.max_len)
        state_cls_embedding = state_embeddings[:, 0]  # Use [CLS] token embedding

        # --- 2. Critic Path ---
        # The critic only needs to evaluate the state's value.
        state_values = self.value_head(state_cls_embedding)

        # --- 3. Actor Path (Hybrid Batching) ---
        # --- Step 3a: Batch embed all actions ---
        all_actions_flat = [a for actions in action_sequences_batch for a in actions]
        action_lengths = [len(actions) for actions in action_sequences_batch]

        if not all_actions_flat:
            # Handle case where there are no actions in the entire batch
            action_scores = [torch.tensor([], device=self.device) for _ in state_sequence_batch]
            return action_scores, state_values
            
        action_texts = [' '.join(map(str, a)) for a in all_actions_flat]
        action_embeddings_all = self.action_encoder(action_texts, max_len=16)
        action_cls_embeddings_flat = action_embeddings_all[:, 0]  # Shape: [total_actions, hidden_size]

        # --- Step 3b: Split flat action embeddings back into a list ---
        action_embeddings_split = list(torch.split(action_cls_embeddings_flat, action_lengths))

        # --- Step 3c: Loop for cross-attention (preserving correct logic) ---
        action_scores = []
        for i in range(len(state_sequence_batch)):
            current_state_cls = state_cls_embedding[i]
            # Use the pre-computed action embeddings
            action_cls_embeddings = action_embeddings_split[i] 

            if action_cls_embeddings.numel() == 0:
                action_scores.append(torch.tensor([], device=self.device))
                continue

            query = current_state_cls.unsqueeze(0).unsqueeze(0)  # Shape: [hidden_size] -> [1, 1, hidden_size]
            key_value = action_cls_embeddings.unsqueeze(
                0)  # Shape: [num_actions, hidden_size] -> [1, num_actions, hidden_size]

            attn_output, _ = self.cross_attention(query=query, key=key_value, value=key_value)

            fused_embeddings = key_value + attn_output
            fused_embeddings = fused_embeddings.squeeze(0)  # Shape: [num_actions, hidden_size]

            # Get scores from the actor head
            scores = self.actor_head(fused_embeddings).squeeze(-1)
            action_scores.append(scores)

        return action_scores, state_values

    def select_action(self, state_sequence, action_sequences):
        """
        Select an action based on the current state and available action sequences,
        and return the chosen action, its log probability, and the value of the state.
        """
        if not action_sequences:
            dummy_action = ('NEXT', 'NEXT')
            dummy_log_prob = torch.tensor(-1e9, device=self.device)  # Log of near-zero probability
            dummy_value = torch.tensor(0.0, device=self.device)
            return dummy_action, dummy_log_prob, dummy_value

        self.eval()
        with torch.no_grad():
            scores, value = self.forward([state_sequence], [action_sequences])

            # If scores are returned as a list containing one tensor
            if isinstance(scores, list):
                scores = scores[0]

            # Ensure scores is a 1D tensor
            if scores.dim() == 0:
                scores = scores.unsqueeze(0)

            # Use softmax to convert scores to probabilities
            probs = F.softmax(scores, dim=-1)

            # Create a categorical distribution and sample from it
            dist = Categorical(probs)
            action_index = dist.sample()

            # Get the chosen action and its log probability
            chosen_action = action_sequences[action_index.item()]
            log_prob = dist.log_prob(action_index)

        self.train()  # Set the model back to training mode
        return chosen_action, log_prob, value

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr, weight_decay=1e-5)


class RNDModule(nn.Module):
    """
    Random Network Distillation (RND) Module for Transformer-based states.
    """
    def __init__(self, input_bit, max_p, max_len, vocab_path='./vocab.json'):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Target network: fixed and randomly initialized
        self.target = TransformerStateEncoder(input_bit, max_p, max_len, vocab_path)
        
        # Predictor network: trained to predict the target's output
        self.predictor = TransformerStateEncoder(input_bit, max_p, max_len, vocab_path)

        # Output heads
        feature_out_dim = 256
        hidden_size = self.target.hidden_size
        self.target_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, feature_out_dim)
        )
        self.predictor_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, feature_out_dim)
        )

        # Freeze the target network
        for param in self.target.parameters():
            param.requires_grad = False
        for param in self.target_head.parameters():
            param.requires_grad = False
    
    def forward(self, state_sequence_batch, max_len):
        """
        Computes the target and predictor features for a batch of state sequences.
        """
        with torch.no_grad():
            target_embeddings = self.target(state_sequence_batch, max_len)
            target_cls = target_embeddings[:, 0] # Use [CLS] token
            target_features = self.target_head(target_cls)

        predictor_embeddings = self.predictor(state_sequence_batch, max_len)
        predictor_cls = predictor_embeddings[:, 0]
        predictor_features = self.predictor_head(predictor_cls)

        return target_features, predictor_features

    def compute_intrinsic_reward(self, state_sequence_batch, max_len):
        """
        Calculates the intrinsic reward for a batch of states.
        """
        with torch.no_grad():
            target_features, predictor_features = self.forward(state_sequence_batch, max_len)
            # The reward is the mean squared error between the predictions and the targets
            reward = F.mse_loss(predictor_features, target_features, reduction='none').mean(dim=-1)
        return reward


if __name__ == '__main__':
    # This test script needs to be updated to match the new env and agent structure
    print("Agent class defined. To test, please run train_ppo.py")