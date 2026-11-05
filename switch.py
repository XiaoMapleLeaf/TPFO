import numpy as np
import random

# 分割函数
def split_into_segments(total, parts):
    if parts == 0:
        return []
    # 随机分割total成parts个正整数
    cuts = np.sort(np.random.choice(range(1, total), parts - 1, replace=False))
    prev = 0
    res = []
    for cut in cuts:
        res.append(cut - prev)
        prev = cut
    res.append(total - prev)
    return res

def construct_array_with_transitions(all_num=100, sp=0.4, trans=0.1, start_num = 0):
    # 确保跳变次数是偶数的一半用于0->1，一半用于1->0
    num_1 = int(all_num * sp)
    #print(num_1)
    num_0 = all_num - num_1
    transitions = int(all_num * trans)
    #print(transitions)
    segments = transitions + 1  # 段数比跳变多1
    #print(transitions)
    
    # 分配段数给0和1
    # 假设以0开头，则0的段数为 (segments + 1) // 2，1的段数为 segments // 2
    if(start_num == 0):
        segments_0 = (segments + 1) // 2
        segments_1 = segments // 2
    else:
        segments_0 = segments // 2
        segments_1 = (segments + 1) // 2
    
    # 将num_0和num_1平均分配到各自的段中
    zeros = split_into_segments(num_0, segments_0)
    #print(zeros)
    ones = split_into_segments(num_1, segments_1)

    # 游程值与长度展开成整段序列，位值与原来的逐段 extend 相同
    values = []
    lengths = []
    i0, i1 = 0, 0
    start_with_zero = segments_0 > segments_1  # 如果0的段数更多，就以0开头

    while i0 < len(zeros) or i1 < len(ones):
        if start_with_zero:
            if i0 < len(zeros):
                values.append(0)
                lengths.append(zeros[i0])
                i0 += 1
            if i1 < len(ones):
                values.append(1)
                lengths.append(ones[i1])
                i1 += 1
        else:
            if i1 < len(ones):
                values.append(1)
                lengths.append(ones[i1])
                i1 += 1
            if i0 < len(zeros):
                values.append(0)
                lengths.append(zeros[i0])
                i0 += 1

    if lengths:
        result = np.repeat(np.asarray(values, dtype=np.uint8), np.asarray(lengths, dtype=np.int64))
    else:
        result = np.zeros(0, dtype=np.uint8)

    # 检查结果是否合法
    # print("num0:{},num1:{},len(result):{}".format(num_0,num_1,len(result)))
    ones_count = int(result.sum(dtype=np.int64))
    assert result.size == num_0 + num_1
    assert result.size - ones_count == num_0
    assert ones_count == num_1

    return result

def ha(a,b):
    a = np.asarray(a, dtype=np.uint8)
    b = np.asarray(b, dtype=np.uint8)
    s = np.bitwise_xor(a, b)
    c = np.bitwise_and(a, b)
    return s, c

def fa(a, b, cin):
    a = np.asarray(a, dtype=np.uint8)
    b = np.asarray(b, dtype=np.uint8)
    cin = np.asarray(cin, dtype=np.uint8)
    axb = np.bitwise_xor(a, b)
    s = np.bitwise_xor(axb, cin)
    c = np.bitwise_or(np.bitwise_and(a, b), np.bitwise_and(axb, cin))
    return s, c

def _ones_and_toggles(signal):
    signal = np.asarray(signal)
    ones = int(signal.sum(dtype=np.int64))
    if signal.size < 2:
        return ones, 0
    toggles = int(np.count_nonzero(signal[1:] != signal[:-1]))
    return ones, toggles

def ha_switch(a_trans,a_sp,b_trans,b_sp,cycle = 100,time = 500,length = 100):
    s_swith = 0
    s_num = 0
    c_switch = 0
    c_num = 0
    num = time * length
    a_trans = a_trans / length
    b_trans = b_trans / length
    for j in range(cycle):
        start_num = random.randint(0, 1)
        try:
            a = construct_array_with_transitions(num,a_sp,a_trans,start_num)
        except:
            a = np.zeros(num, dtype=np.uint8)
        start_num = random.randint(0, 1)
        try:
            b = construct_array_with_transitions(num,b_sp,b_trans,start_num)
        except:
            b = np.zeros(num, dtype=np.uint8)
        s, c = ha(a,b)
        s_ones, s_toggles = _ones_and_toggles(s)
        c_ones, c_toggles = _ones_and_toggles(c)
        s_num += s_ones
        c_num += c_ones
        s_swith += s_toggles
        c_switch += c_toggles
    s_trans = s_swith / (cycle * time)
    c_trans = c_switch / (cycle * time)
    s_sp = s_num / (cycle * num)
    c_sp = c_num / (cycle * num)
    return s_trans, s_sp, c_trans, c_sp

def fa_switch(a_trans,a_sp,b_trans,b_sp,cin_trans,cin_sp,cycle = 100,time = 500,length = 100):#time指周期长度，一周期的长度为60
    s_swith = 0
    s_num = 0
    c_switch = 0
    c_num = 0
    num = time * length
    a_trans = a_trans / length
    b_trans = b_trans / length
    cin_trans = cin_trans / length
    for j in range(cycle):
        start_num = random.randint(0, 1)
        try:
            a = construct_array_with_transitions(num,a_sp,a_trans,start_num)
        except:
            a = np.zeros(num, dtype=np.uint8)
        start_num = random.randint(0, 1)
        try:
            b = construct_array_with_transitions(num,b_sp,b_trans,start_num)
        except:
            b = np.zeros(num, dtype=np.uint8)
        start_num = random.randint(0, 1)
        try:
            cin = construct_array_with_transitions(num,cin_sp,cin_trans,start_num)
        except:
            cin = np.zeros(num, dtype=np.uint8)
        s, c = fa(a,b,cin)
        s_ones, s_toggles = _ones_and_toggles(s)
        c_ones, c_toggles = _ones_and_toggles(c)
        s_num += s_ones
        c_num += c_ones
        s_swith += s_toggles
        c_switch += c_toggles
    s_trans = s_swith / (cycle * time)
    c_trans = c_switch / (cycle * time)
    s_sp = s_num / (cycle * num)
    c_sp = c_num / (cycle * num)
    return s_trans, s_sp, c_trans, c_sp        

def main():
    a_trans = 0
    a_sp = 0
    b_trans = 0.1
    b_sp = 0.5
    cin_trans = 0.1
    cin_sp = 0.5
    s_trans, s_sp, c_trans, c_sp = ha_switch(a_trans,a_sp,b_trans,b_sp)
    print("a_trans:{}".format(a_trans), "a_sp:{}".format(a_sp))
    print("b_trans:{}".format(b_trans), "b_sp:{}".format(b_sp))
    print("s_trans:{}".format(s_trans), "s_sp:{}".format(s_sp))
    print("c_trans:{}".format(c_trans), "c_sp:{}".format(c_sp))
    s_trans, s_sp, c_trans, c_sp = fa_switch(a_trans,a_sp,b_trans,b_sp,cin_trans,cin_sp)
    print(s_trans, s_sp)
    print(c_trans, c_sp)

if __name__ == "__main__":
    main()
