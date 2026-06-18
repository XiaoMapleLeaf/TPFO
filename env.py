import sys

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from itertools import combinations
import random
from tqdm import tqdm
from readv_enhanced_optimized import *
from delay import cell_area, cell_delay
from switch import fa_switch, ha_switch
import time
import os
import hashlib
from utils import Regression
import subprocess
from diskcache import Cache
import shutil

FAcell = "AD1V2C_140P9T35R"
HAcell1 = "CLKAND2V2_140P9T35R"
HAcell2 = "OA1B2V2_140P9T35R"

oprpath = os.path.expanduser('~')


def compute_dp(rdelay, rt, power, static, delay_w, power_w):
    dp = (rdelay + rt) * delay_w + (0.2 * power + static) * power_w
    return dp


class CircuitEnv(gym.Env):
    """
    A circuit generation environment designed for the Transformer, using the "delay final reward" strategy.
    """

    def __init__(self, input_bit=4, verilog=None, dc=False, archive_manager=None):
        super().__init__()
        self.input_bit = input_bit
        self.output_bit = 2 * input_bit - 1
        self.dc = dc
        self.archive_manager = archive_manager
        self.current_trajectory = []
        self.delay_w = 1.0
        self.area_w = 0.01
        self.power_w = 1.0
        self.best_r = -sys.maxsize
        self.best_v = None
        self.best_delay = float('inf')
        self.best_delay_v = None
        self.best_power = float('inf')
        self.best_power_v = None
        self.best_delay_file = None
        self.best_power_file = None
        self.best_reward_file = None
        self.init_verilog = verilog
        self.step_verilog = verilog
        self.steps = 0
        self.initial_power = None
        self.initial_delay = None
        self.initial_area = None
        self.power_at_stage_start = None
        self.DELAY_CONSTRAINT = 800.0
        self.DELAY_SAFE_ZONE_END = 790.0
        self.BUFFER_DELAY_LIMIT = 810.0
        self.DELAY_PROXIMITY_BONUS_COEFF = 0.5
        self.DELAY_LINEAR_PENALTY_COEFF = 1.0
        self.DELAY_QUADRATIC_PENALTY_COEFF = 0.5
        self.power_reward_scaling_factor = (self.input_bit ** 2) / 16
        
        self.steps_in_current_stage = 0
        self.max_steps_per_stage = 2 * self.input_bit # Allow up to 'input_bit' swaps per stage
        self.top_k_swaps = 10

        self.current_d = float('inf')
        self.current_p = float('inf')
        self.current_a = float('inf')
        self.dones = 0
        self.one_stage = 0
        self.signals = None
        self.connections = None
        self.previous_power = None


        self.action_space = spaces.Discrete(3)

        self.observation_space = spaces.Dict({
            "state_sequence": spaces.Text(max_length=1024),
            "action_sequences": spaces.Sequence(spaces.Text(max_length=256))
        })

        CACHE_DIR = "verilog_analysis_cache"
        self.result_cache = Cache(CACHE_DIR)

        self.reset()

    def save_cache(self):
        """Closes the disk cache gracefully."""
        try:
            self.result_cache.close()
        except Exception as e:
            tqdm.write(f"❌ Error closing cache: {e}")

    def _run_and_cache_estimate(self, verilog_path):
        """Wrapper for estimate_readv with diskcache caching."""
        try:
            with open(verilog_path, 'rb') as f:
                file_content = f.read()
            verilog_hash = hashlib.sha256(file_content).hexdigest()
        except (IOError, TypeError):
            verilog_hash = str(verilog_path)

        try:
            return self.result_cache[verilog_hash]
        except KeyError:
            d_raw, p_raw, swapping, signals, fas, has, connections = estimate_readv(verilog_path, self.input_bit, self.dc)
            result = (d_raw, p_raw, swapping, signals, fas, has, connections)
            self.result_cache[verilog_hash] = result
            return result

    def reset(self, seed=None, options=None, start_from_milestone=None):
        super().reset(seed=seed)

        if start_from_milestone is not None:
            source_verilog_path = start_from_milestone['verilog_path']
            verilog_hash = start_from_milestone['verilog_hash']
            
            new_filename = f"multiplier_{self.input_bit}b_temp.v"
            destination_verilog_path = os.path.join('run_verilog_mult_mid', new_filename)
            
            try:
                shutil.copy2(source_verilog_path, destination_verilog_path)
                self.step_verilog = destination_verilog_path
            except (FileNotFoundError, Exception) as e:
                tqdm.write(f"❌ 错误: 恢复里程碑文件 '{source_verilog_path}' 失败: {e}。将重置到初始状态。")
                start_from_milestone = None

        if start_from_milestone is not None:
            env_state = start_from_milestone['env_state']
            
            self.steps_in_current_stage = env_state['steps_in_current_stage']
            self.initial_power = env_state['initial_power']
            self.initial_delay = env_state['initial_delay']
            self.power_at_stage_start = env_state['power_at_stage_start']
            self.dones = env_state.get('dones', 0)
            self.one_stage = env_state.get('one_stage', 0)
            self.current_trajectory = start_from_milestone.get('trajectory_to_state', [])
        else:
            self.step_verilog = self.init_verilog
            self.steps_in_current_stage = 0
            self.dones = 0
            self.one_stage = 0
            self.current_trajectory = []
        
        d_raw, p_raw, valid_actions, self.signals, fas, has, self.connections = self._run_and_cache_estimate(self.step_verilog)
        
        afa = cell_area(FAcell)
        aha1 = cell_area(HAcell1)
        aha2 = cell_area(HAcell2)
        a_calc = fas * afa + has * (aha1 + aha2)
        self.current_d = d_raw * 1000
        self.current_a = a_calc / 10
        self.current_p = p_raw * 1000
        
        self.previous_power = self.current_p
        self.valid_actions = list(valid_actions.items())

        if not start_from_milestone:
            self.initial_power = self.current_p
            self.initial_delay = self.current_d
            self.initial_area = self.current_a
            self.power_at_stage_start = self.current_p
            if self.best_delay == float('inf') or self.best_delay is None: self.best_delay = self.current_d
            if self.best_power == float('inf') or self.best_power is None: self.best_power = self.current_p
        self.start_stage_index = self._find_first_meaningful_stage()
        self.steps = self.start_stage_index

        return self._get_obs(), self._get_info()

    def _find_first_meaningful_stage(self):
        """
        Find the first meaningful optimization stage, skipping early symmetric stages.
        """
        for i, (_col, signals) in enumerate(self.valid_actions):
            has_low_toggle = False
            has_high_toggle = False
            has_ip_signal = False
            has_p_signal = False

            for signal in signals:
                if signal.startswith('ip_'):
                    has_ip_signal = True
                    try:
                        parts = signal.split('_')
                        i_index = int(parts[1])
                        if i_index < self.input_bit / 2:
                            has_low_toggle = True
                        else:
                            has_high_toggle = True
                    except (IndexError, ValueError):
                        continue
                elif signal.startswith('pin'):
                    has_p_signal = True

            if (has_low_toggle and has_high_toggle) or (has_ip_signal and has_p_signal):
                tqdm.write(f"找到第一个有意义的优化阶段: Stage {i}")
                return i
        
        tqdm.write("未找到混合信号阶段，将从 Stage 0 开始优化。")
        return 0

    def _calculate_reward(self, improvement, base_value, penalty_coefficient=1.5, dead_zone_threshold=0.01):
        """
        Calculate reward based on the percentage of power performance improvement, including dead zone and asymmetric penalty.
        :param improvement: Performance improvement value (e.g., old_power - new_power)
        :param base_value: Base value for calculating the percentage (e.g., old_power)
        :param penalty_coefficient: Penalty coefficient for negative improvement
        :param dead_zone_threshold: Threshold for the reward dead zone (e.g., 0.001 represents 0.1%)
        :return: Calculated reward value
        """
        if abs(base_value) < 1e-9:
            return 0.0

        percentage_change = improvement / base_value

        # 1. Reward dead zone: Ignore small noise fluctuations
        if abs(percentage_change * self.power_reward_scaling_factor) < dead_zone_threshold:
            return 0.0

        # 2. Asymmetric penalty: Apply a more severe penalty for performance deterioration
        if percentage_change < 0:
            return percentage_change * 100 * penalty_coefficient * self.power_reward_scaling_factor

        # 3. Performance improvement reward
        return percentage_change * 100 * self.power_reward_scaling_factor

    def _calculate_delay_reward_component(self, current_delay):
        """
        Calculate the four-part reward/penalty component based on the timing constraint.
        - Safe zone (<= 490ns): Reward is 0.
        - Incentive zone (490-500ns): Linear positive reward.
        - Penalty zone (500-510ns): Linear penalty.
        - Timeout zone (> 510ns): Severe quadratic penalty.
        """
        if current_delay <= self.DELAY_SAFE_ZONE_END:
            # Safe zone: No reward
            return 0.0

        elif current_delay <= self.DELAY_CONSTRAINT:
            # Incentive zone: Reward grows linearly from 0 to the maximum reward value
            # Calculate the current delay in the incentive zone (0 to 1)
            proximity_score = (current_delay - self.DELAY_SAFE_ZONE_END) / (self.DELAY_CONSTRAINT - self.DELAY_SAFE_ZONE_END)
            bonus = proximity_score * self.DELAY_PROXIMITY_BONUS_COEFF
            return bonus

        elif current_delay <= self.BUFFER_DELAY_LIMIT:
            # Buffer zone: Linear penalty
            violation = current_delay - self.DELAY_CONSTRAINT
            return -violation * self.DELAY_LINEAR_PENALTY_COEFF
        
        else: # current_delay > self.BUFFER_DELAY_LIMIT
            # Timeout zone: Severe quadratic penalty
            # Penalty is continuous at the boundary
            base_penalty = -(self.BUFFER_DELAY_LIMIT - self.DELAY_CONSTRAINT) * self.DELAY_LINEAR_PENALTY_COEFF
            extra_violation = current_delay - self.BUFFER_DELAY_LIMIT
            quadratic_penalty = extra_violation ** 2 * self.DELAY_QUADRATIC_PENALTY_COEFF
            return base_penalty - quadratic_penalty

    def _score_a_single_swap(self, s1, s2):
        """
        Calculate the "homogeneity improvement" score for a single swap action (s1, s2).
        Score = (std_dev_before - std_dev_after) * coeff + (FA c-pin pw improvement) * coeff
        Positive score represents optimization.
        """
        if not self.connections or s1 == s2:
            return 0.0

        # Helper function to get the pw value of a signal (power proxy)
        def get_pw(signal_name):
            phys = self.signals.get(signal_name, {})
            print(f"{signal_name}: {(phys.get('rise_transition', 0), phys.get('fall_transition', 0), phys.get('toggle', 0), phys.get('static', 0))}")
            return compute_dp(phys.get('rise_delay', 0), phys.get('rise_transition', 0), 
                              phys.get('toggle', 0), phys.get('static', 0), 0, self.power_w)

        # 1. Find all affected components by this swap
        affected_components = {} # {comp_id: [inputs]}
        
        # Find the load components of s1 and s2
        affected_comp_ids = set()
        if s1 in self.connections.get('signal_to_loads', {}):
            for load in self.connections['signal_to_loads'][s1]:
                affected_comp_ids.add(f"{load['type']}_{load['instance_id']}")
        if s2 in self.connections.get('signal_to_loads', {}):
            for load in self.connections['signal_to_loads'][s2]:
                affected_comp_ids.add(f"{load['type']}_{load['instance_id']}")
        print(s1)
        print(s2)
        print(affected_comp_ids)

        if not affected_comp_ids:
            return 0.0

        # 2. Get the input list of these components from the global connections
        fa_list = self.connections.get('fa_instances', [])
        ha_list = self.connections.get('ha_instances', [])
        all_components = fa_list + ha_list
        for comp in all_components:
            comp_type = 'FA' if len(comp['inputs']) == 3 else 'HA'
            comp_id = f"{comp_type}_{comp['id']}"
            if comp_id in affected_comp_ids:
                affected_components[comp_id] = comp
        
        # --- NEW: Debugging logic to print output loads ---
        for comp_id, comp_info in affected_components.items():
            print(f"--- Analyzing loads for {comp_id} ---")
            for output_signal in comp_info['outputs']:
                load_cap = calculate_load_capacitance(output_signal, self.connections)
                print(f"  -> Output '{output_signal}': Load Capacitance (Rise/Fall) = {load_cap}")
        # --- END NEW ---

        # 3. Calculate the total "homogeneity improvement" score for all affected components by this swap
        total_homogeneity_improvement = 0
        total_pin_preference_improvement = 0 # NEW
        for comp_id, comp_info in affected_components.items():
            original_inputs = comp_info['inputs']
            # Ensure this component is indeed a load of s1 or s2
            if s1 not in original_inputs and s2 not in original_inputs:
                continue
                
            # Part 1: Homogeneity Score (Power Heuristic)
            pws_before = [get_pw(s) for s in original_inputs]
            print(f"pws_before: {pws_before}")
            std_dev_before = np.std(pws_before) if len(pws_before) > 1 else 0
            print(f"std_dev_before: {std_dev_before}")

            swapped_inputs = [s2 if s == s1 else (s1 if s == s2 else s) for s in original_inputs]
            
            pws_after = [get_pw(s) for s in swapped_inputs]
            print(f"pws_after: {pws_after}")
            std_dev_after = np.std(pws_after) if len(pws_after) > 1 else 0
            print(f"std_dev_after: {std_dev_after}")
            total_homogeneity_improvement += (std_dev_before - std_dev_after)
            print(f"total_homogeneity_improvement: {total_homogeneity_improvement}")

            # Part 2: FA Pin Preference Score (Delay Heuristic)
            if comp_id.startswith('FA') and len(original_inputs) == 3:
                # original_inputs is ordered [a, b, c]
                pw_c_before = get_pw(original_inputs[2])
                print(f"pw_c_before: {pw_c_before}")
                pw_c_after = get_pw(swapped_inputs[2])
                print(f"pw_c_after: {pw_c_after}")
                # The "improvement" is the increase of the power value on pin c
                total_pin_preference_improvement += (pw_c_after - pw_c_before)
                print(f"total_pin_preference_improvement: {total_pin_preference_improvement}")

        # Multiply by a coefficient to make the reward scale more appropriate
        homogeneity_coeff = 10.0
        pin_preference_coeff = 0.0 # Heuristic weight for the new rule

        final_score = (total_homogeneity_improvement * homogeneity_coeff) + \
                      (total_pin_preference_improvement * pin_preference_coeff)
                      
        return final_score

    def step(self, index):
        valid_action = self._generate_valid_actions(self.steps)
        pos = valid_action[index] if index < len(valid_action) else ('NEXT', 'NEXT')
        # print(f"pos: {pos}")

        truncated = False

        # --- RE-IMPLEMENTED: Budgeted Action Logic ---
        if pos[0] == 'NEXT':
            # Agent chooses to advance to the next stage
            self.steps += 1
            self.steps_in_current_stage = 0 # Reset budget counter for the new stage
        elif pos[0] != 'IDLE':
            # A SWAP action was chosen, consuming budget
            self.steps_in_current_stage += 1
            if self.step_verilog is None:
                self.step_verilog = swap_in_stage(self.init_verilog, pos[0], pos[1], self.dones, self.input_bit)
            else:
                self.step_verilog = swap_in_stage(self.step_verilog, pos[0], pos[1], self.dones, self.input_bit)
        
        # --- Go-Explore: 记录轨迹 ---
        self.current_trajectory.append({'action': pos, 'reward': 0}) # reward稍后更新

        if self.steps_in_current_stage >= self.max_steps_per_stage:
            tqdm.write(f"--- Budget exhausted ({self.max_steps_per_stage} steps), forcing stage transition ---")
            self.steps += 1
            self.steps_in_current_stage = 0
            pos = ('NEXT', 'NEXT')

        is_last_stage = self.steps >= len(self.valid_actions) -1
        done = (pos[0] == 'NEXT' and is_last_stage) or self.steps >= len(self.valid_actions)
        # --- END REVISED ---
        
        # --- STATE UPDATE & REWARD CALCULATION ---
        d_raw, p_raw, swapping, signals, fas, has, self.connections = self._run_and_cache_estimate(self.step_verilog)

        # FIX 3: Robustly update state from the ground truth (the file analysis)
        self.signals = signals
        self.valid_actions = list(swapping.items())
        # print(f"self.valid_actions: {self.valid_actions}")

        # Calculate PPA from raw values
        afa = cell_area(FAcell)
        aha1 = cell_area(HAcell1)
        aha2 = cell_area(HAcell2)
        a_calc = fas * afa + has * (aha1 + aha2)
        d = d_raw * 1000
        a = a_calc / 10
        p = p_raw * 1000

        # --- NEW: Update previous_power for print analysis ---
        self.previous_power = self.current_p
        # --- END NEW ---
        # --- NEW: Update current global PPA after step ---
        self.current_d = d
        self.current_a = a
        self.current_p = p
        # --- END NEW ---

        obs_for_return = self._get_obs()

        # --- REVISED: Greedy Heuristic Reward Calculation ---
        power_reward = 0
        
        # 1. For SWAP actions, the reward equals the "homogeneity improvement score"
        if pos[0] not in ['IDLE', 'NEXT']:
            step_improvement = self.previous_power - self.current_p
            power_reward = self._calculate_reward(step_improvement, self.previous_power)
        elif pos[0] == 'NEXT':
            stage_improvement = self.power_at_stage_start - self.current_p
            power_reward = self._calculate_reward(stage_improvement, self.power_at_stage_start)
            # Update the power baseline for the next stage
            self.power_at_stage_start = self.current_p

        # 2. If the episode ends, add an additional reward based on the total power improvement
        if done or truncated:
            total_improvement = self.initial_power - self.current_p
            final_power_reward = self._calculate_reward(total_improvement, self.initial_power)
            power_reward += final_power_reward
            # power_reward = self._score_a_single_swap(pos[0], pos[1])
        # 3. Calculate the reward/penalty component based on the timing constraint
        delay_reward_component = self._calculate_delay_reward_component(self.current_d)
        
        # 3. Combine into the final reward
        reward = power_reward + delay_reward_component
        # --- END REVISED ---

        # --- Go-Explore: Update the reward in the trajectory ---
        if self.current_trajectory:
            self.current_trajectory[-1]['reward'] = reward
        
        if self.steps % 100 == 0:
            self._log_step_analysis(reward, power_reward, delay_reward_component)

        # Update bests and check if we need to save the circuit file
        need_save = False
        # The metric for "best reward" should be the total improvement, not the step reward
        total_improvement_metric = self.initial_power - self.current_p

        # if self.current_d < self.best_delay:
        #     self.best_delay = d
        #     self.best_delay_v = self.step_verilog
        #     need_save = True
        # improvement = self.previous_power - self.current_p
        if self.current_p < self.best_power: # or improvement > 1.5:
            # if self.current_p < self.best_power:
            self.best_power = p
            self.best_power_v = self.step_verilog
            need_save = True
            
            # --- Go-Explore: Immediately archive when a new global best power is found ---
            if self.archive_manager is not None:
                try:
                    with open(self.step_verilog, 'rb') as f:
                        file_content = f.read()
                    verilog_hash = hashlib.sha256(file_content).hexdigest()
                    
                    milestone_path = os.path.join(self.archive_manager.archive.directory, f"{verilog_hash}.v")
                    env_state_to_save = self.get_milestone_state(milestone_path)

                    milestone_data = {
                        "verilog_hash": verilog_hash,
                        "verilog_path": milestone_path,
                        "achieved_power": self.current_p,
                        "achieved_delay": self.current_d,
                        "env_state": env_state_to_save,
                        "trajectory_to_state": self.current_trajectory
                    }
                    self.archive_manager.add_milestone(milestone_data)
                    tqdm.write(f"💾 Save elite milestone: {milestone_path} (Power: {self.current_p:.2f})")
                except Exception as e:
                    tqdm.write(f"❌ Save elite milestone failed: {e}")
            # --- End Go-Explore ---


        self.dones += 1

        if need_save:
            self.save_best_files()

        # print(f"done: {done}, truncated: {truncated}")
        # The 'truncated' variable is now meaningful again.
        if done or truncated:
            return obs_for_return, reward, done, truncated, self._get_info()

        return obs_for_return, reward, done, truncated, self._get_info()

    def _score_swaps_by_homogeneity(self, potential_swaps, current_step):
        """
        Score and sort all possible swap actions based on the prior knowledge of "pw values clustering".
        Return the top top_k_swaps actions.
        """
        if not self.connections or not potential_swaps:
            return [tuple(sorted(s)) for s in potential_swaps]

        # 1. Identify all components in the current stage and build a mapping from component ID to its inputs
        component_inputs = {}
        _col, signals_in_stage = self.valid_actions[current_step]
        stage_comp_ids = set()
        for signal in signals_in_stage:
            if signal in self.connections.get('signal_to_sources', {}):
                source = self.connections['signal_to_sources'][signal]
                stage_comp_ids.add(f"{source['type']}_{source['instance_id']}")
            if signal in self.connections.get('signal_to_loads', {}):
                for load in self.connections['signal_to_loads'][signal]:
                    stage_comp_ids.add(f"{load['type']}_{load['instance_id']}")

        for fa in self.connections.get('fa_instances', []):
            comp_id = f"FA_{fa['id']}"
            if comp_id in stage_comp_ids:
                component_inputs[comp_id] = fa['inputs']
        for ha in self.connections.get('ha_instances', []):
            comp_id = f"HA_{ha['id']}"
            if comp_id in stage_comp_ids:
                component_inputs[comp_id] = ha['inputs']

        # Helper function to get the pw value of a signal
        def get_pw(signal_name):
            phys = self.signals.get(signal_name, {})
            return compute_dp(phys.get('rise_delay', 0), phys.get('rise_transition', 0), 
                              phys.get('toggle', 0), phys.get('static', 0), 0, self.power_w)

        # 2. Score each potential swap action
        scored_swaps = []
        for s1, s2 in potential_swaps:
            total_score_improvement = 0
            
            # Find all affected components by this swap
            affected_comp_ids = set()
            if s1 in self.connections.get('signal_to_loads', {}):
                for load in self.connections['signal_to_loads'][s1]:
                    affected_comp_ids.add(f"{load['type']}_{load['instance_id']}")
            if s2 in self.connections.get('signal_to_loads', {}):
                for load in self.connections['signal_to_loads'][s2]:
                    affected_comp_ids.add(f"{load['type']}_{load['instance_id']}")

            if not affected_comp_ids:
                scored_swaps.append(((s1, s2), 0))
                continue

            # Calculate the total "homogeneity improvement" score for all affected components by this swap
            for comp_id in affected_comp_ids:
                if comp_id not in component_inputs:
                    continue
                
                original_inputs = component_inputs[comp_id]
                
                pws_before = [get_pw(s) for s in original_inputs]
                std_dev_before = np.std(pws_before) if len(pws_before) > 1 else 0

                swapped_inputs = [s2 if s == s1 else (s1 if s == s2 else s) for s in original_inputs]
                
                pws_after = [get_pw(s) for s in swapped_inputs]
                std_dev_after = np.std(pws_after) if len(pws_after) > 1 else 0
                
                total_score_improvement += (std_dev_before - std_dev_after)

            scored_swaps.append(((s1, s2), total_score_improvement))

        # 3. Sort by score in descending order
        scored_swaps.sort(key=lambda x: x[1], reverse=True)
        
        # 4. Return the top top_k_swaps actions
        pruned_swaps = [tuple(sorted(swap)) for swap, score in scored_swaps[:self.top_k_swaps]]
        # print(f"pruned_swaps: {pruned_swaps}")
        # print(f"scored_swaps: {scored_swaps[:self.top_k_swaps]}")
        return pruned_swaps

    def _log_step_analysis(self, total_reward, power_reward, delay_reward):
        """Logs the analysis of the current step, including reward components."""
        improvement = 0
        if self.previous_power is not None:
            improvement = self.previous_power - self.current_p

        tqdm.write(
            f"\n[ANALYSIS] Circuit generated. Delay: {self.current_d:.2f}, Area: {self.current_a:.2f}, Power: {self.current_p:.2f}, Step Improvement: {improvement:.2f}, "
            f"Reward: {total_reward:.4f} (Power: {power_reward:.2f}, Delay: {delay_reward:.2f}), self.step_verilog: {self.step_verilog}"
        )

    def _get_state_sequence_for_stage(self, stage_index):
        """
        Generate the local, topology-related text sequence description for a specified single stage.
        格式: COL <c> DOTS <d> SIG <n1> dly <path_delay> pwr <power_metric> ...
        """
        # Safety check to ensure the stage index is valid
        if stage_index >= len(self.valid_actions):
            return ""

        col, available_signals = self.valid_actions[stage_index]
        dots = len(available_signals)

        # Only generate local information
        state_str = f"COL {col} DOTS {dots} "

        signal_strings = []
        for s_name in available_signals:
            if s_name in self.signals:
                phys = self.signals.get(s_name)
                signal_strings.append(
                    f"SIG {s_name} "
                    f"dl {compute_dp(phys['rise_delay'], phys['rise_transition'], phys['toggle'], phys['static'], self.delay_w, 0):.3f} "
                    f"pw {compute_dp(phys['rise_delay'], phys['rise_transition'], phys['toggle'], phys['static'], 0, self.power_w):.3f} "
                )

        return (state_str + " ".join(signal_strings)).strip()

    def _get_obs(self):
        # 1. Generate one global PPA information
        global_ppa_str = (
            f"G_PWR {self.current_p:.2f} G_DLY {self.current_d:.2f} "
            f"G_DLY_CONSTRAINT {self.DELAY_CONSTRAINT:.2f}"
        )
        
        # 2. Generate the "focus" part: the detailed topology of the current active stage, and re-inject the step information
        active_stage_topology = self._get_state_sequence_for_stage(self.steps)
        if active_stage_topology:
            step_info = f"ACTIVE_STAGE STEP {self.steps_in_current_stage} LIMIT {self.max_steps_per_stage}"
            parts = active_stage_topology.split(' ')
            parts.insert(4, step_info) 
            active_stage_topology = " ".join(parts)

        # 3. Generate the context
        full_topology_parts = []
        # for i in range(self.steps-1, self.steps, 1):
        for i in range(len(self.valid_actions)):
            stage_topology = self._get_state_sequence_for_stage(i)
            if stage_topology:
                full_topology_parts.append(stage_topology)
        full_topology_str = " ".join(full_topology_parts)
        
        # 4. Assemble the final sequence, using [SEP] to separate the focus and context
        full_sequence = f"CLS {global_ppa_str} {active_stage_topology} SEP {full_topology_str} SEP"

        # 5. Generate the valid actions for the current stage
        valid_actions = self._generate_valid_actions(self.steps)

        return {
            "state_sequence": full_sequence,
            "action_sequences": valid_actions
        }

    def _get_info(self):
        return {
            "best_v": self.best_v,
            "best_r": self.best_r,
            "best_delay_v": self.best_delay_v,
            "best_delay": self.best_delay,
            "best_power_v": self.best_power_v,
            "best_power": self.best_power,
            "current_delay": self.current_d,
            "current_power": self.current_p
        }

    def get_milestone_state(self, new_verilog_path):
        """Save the current verilog file and return the environment state dictionary for archiving."""
        shutil.copy2(self.step_verilog, new_verilog_path)

        return {
            "steps": self.steps,
            "steps_in_current_stage": self.steps_in_current_stage,
            "initial_power": self.initial_power,
            "initial_delay": self.initial_delay,
            "power_at_stage_start": self.power_at_stage_start,
            "dones": self.dones,
            "one_stage": self.one_stage
        }

    def _generate_valid_actions(self, current_step):
        if current_step >= len(self.valid_actions):
            return [] # Return empty list if the episode should be over

        _col, able_items = self.valid_actions[current_step]

        if able_items and self.connections:
            device_ids = set()
            stage_component_types = set()
            for signal in able_items:
                if signal in self.connections.get('signal_to_sources', {}):
                    source = self.connections['signal_to_sources'][signal]
                    device_ids.add(f"{source['type']}_{source['instance_id']}")
                    stage_component_types.add(source['type'])

            if len(device_ids) == 1 and 'HA' in stage_component_types:
                return [('NEXT', 'NEXT')]
        
        # 1. Generate all possible, physically meaningful swap actions
        physically_meaningful_swaps = []
        sorted_able_items = sorted(able_items)
        potential_swaps = combinations(sorted_able_items, 2)

        for comb in potential_swaps:
            signal1_name, signal2_name = comb
            should_add = False
            s1_found = signal1_name in self.signals
            s2_found = signal2_name in self.signals

            if s1_found and s2_found:
                phys1 = self.signals[signal1_name]
                phys2 = self.signals[signal2_name]
                dl1 = compute_dp(phys1['rise_delay'], phys1['rise_transition'], phys1['toggle'], phys1['static'], self.delay_w, 0)
                pw1 = compute_dp(phys1['rise_delay'], phys1['rise_transition'], phys1['toggle'], phys1['static'], 0, self.power_w)
                dl2 = compute_dp(phys2['rise_delay'], phys2['rise_transition'], phys2['toggle'], phys2['static'], self.delay_w, 0)
                pw2 = compute_dp(phys2['rise_delay'], phys2['rise_transition'], phys2['toggle'], phys2['static'], 0, self.power_w)
                if not (np.isclose(dl1, dl2) and np.isclose(pw1, pw2)):
                    should_add = True
            
            if should_add:
                physically_meaningful_swaps.append(tuple(sorted(comb)))

        # 2. Intelligent pruning: if the action space is too large, use prior knowledge to prune
        if self.input_bit >= 4 and len(physically_meaningful_swaps) > self.top_k_swaps:
            # tqdm.write(f"  -> Stage {current_step}: Action space large ({len(physically_meaningful_swaps)}). Applying intelligent pruning to Top-{self.top_k_swaps}.")
            final_valid_swaps = self._score_swaps_by_homogeneity(physically_meaningful_swaps, current_step)
        else:
            final_valid_swaps = physically_meaningful_swaps

        # 3. Assemble the final action list
        valid_action = final_valid_swaps
        valid_action.append(('NEXT', 'NEXT'))

        return valid_action

    def save_best_files(self):
        """Save the best files to disk in real-time, overwriting the old best files"""
        import shutil
        import os

        save_dir = "best_circuits"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if self.best_power_v and self.best_power != float('inf'):
            power_filename = f"best_power_{self.input_bit}bit.v"
            power_filepath = os.path.join(save_dir, power_filename)
            try:
                shutil.copy2(self.best_power_v, power_filepath)
                self.best_power_file = power_filepath
                tqdm.write(f"💾 Save best power: {power_filepath} (Power: {self.best_power:.2f})")
            except Exception as e:
                tqdm.write(f"❌ Save power file failed: {e}")

    def get_best_files_summary(self):
        """Get the summary of the best files"""
        summary = {
            'delay': {
                'value': self.best_delay,
                'file': self.best_delay_file,
                'verilog': self.best_delay_v
            },
            'power': {
                'value': self.best_power,
                'file': self.best_power_file,
                'verilog': self.best_power_v
            },
            'reward': {
                'value': self.best_r,
                'file': self.best_reward_file,
                'verilog': self.best_v
            }
        }
        return summary


if __name__ == '__main__':
    env = CircuitEnv(input_bit=8, verilog='run_verilog_mult_mid/multiplier_8b_8a37687a7776b5c7f40ed030b4890e7f.v')
    obs, info = env.reset()
    print("--- Initial State ---")
    print("Sequence:", obs['state_sequence'])

    done = False
    total_reward = 0
    while not done:
        num_actions = len(obs['action_sequences'])
        action_index = random.randint(0, num_actions - 1)
        action_sequences = obs['action_sequences']
        action_taken = action_sequences[action_index]

        print(f"\nStep {env.steps}: Taking action {action_taken}")
        obs, reward, done, truncated, info = env.step(action_index)

        print(f"Intermediate Reward: {reward}")
        if not done:
            print("New Sequence:", obs['state_sequence'])

        if done:
            print(f"\n--- Episode Finished ---")
            print(f"Final Info: {info}")
            print(f"Final (True) Reward: {reward}")