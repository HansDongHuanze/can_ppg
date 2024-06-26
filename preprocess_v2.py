# dataset from https://github.com/MahdiFarvardin/MTVital
import numpy as np
import neurokit2 as nk
import heartpy as hp
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import CubicSpline

path = '/home/dhz/MTVital/MTHS/Data/signal_'


def cubic_transform(seq, new_length):
    original_length = len(seq)
    indices = np.linspace(0, original_length - 1, num=new_length)

    # Create a matrix to store the accumulated distances
    dtw_matrix = np.zeros((new_length, original_length))

    # Compute the accumulated distances
    for i in range(new_length):
        for j in range(original_length):
            cost = abs(indices[i] - j)
            dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j] if i > 0 else float('inf'),
                                          dtw_matrix[i, j-1] if j > 0 else float('inf'),
                                          dtw_matrix[i-1, j-1] if i > 0 and j > 0 else float('inf'))

    # Find the optimal path through the accumulated distances
    path = [(new_length - 1, original_length - 1)]
    i = new_length - 1
    j = original_length - 1

    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            prev_cost = min(dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1])
            if dtw_matrix[i-1, j-1] == prev_cost:
                i -= 1
                j -= 1
            elif dtw_matrix[i-1, j] == prev_cost:
                i -= 1
            else:
                j -= 1
        path.append((i, j))

    path.reverse()

    # Interpolate the values based on the optimal path using cubic spline interpolation
    new_seq = []
    x = np.array(range(original_length))
    y = np.array(seq)
    cs = CubicSpline(x, y)

    for i, j in path:
        if i == new_length - 1:
            new_seq.append(seq[j])
        else:
            start = indices[i]
            end = indices[i + 1]
            new_values = cs(np.linspace(start, end, num=int(end - start) + 1))
            new_seq.extend(new_values)

    return new_seq

def normalization(data):
    """
    归一化函数
    把所有数据归一化到[0，1]区间内，数据列表中的最大值和最小值分别映射到1和0，所以该方法一定会出现端点值0和1。
    此映射是线性映射，实质上是数据在数轴上等比缩放。
    
    :param data: 数据列表，数据取值范围：全体实数
    :return:
    """
    min_value = min(data)
    max_value = max(data)
    new_list = []
    for i in data:
        new_list.append((i-min_value) / (max_value-min_value))
    return new_list

def interpolate_tensor(input_tensor, target_length):
    """
    Interpolates the input tensor to the target length using linear interpolation.
    
    Args:
        input_tensor (torch.Tensor): The input one-dimensional tensor.
        target_length (int): The desired length of the output tensor.
        
    Returns:
        torch.Tensor: The interpolated tensor of the target length.
    """
    input_length = input_tensor.size(0)
    
    # Convert input tensor to NumPy array
    input_np = input_tensor.numpy()
    
    # Create interpolation indices
    input_indices = np.linspace(0, input_length - 1, input_length)
    target_indices = np.linspace(0, input_length - 1, target_length)
    
    # Interpolate using indices
    interpolated_np = np.interp(target_indices, input_indices, input_np)
    
    # Convert back to a PyTorch tensor
    interpolated_tensor = torch.tensor(interpolated_np)
    
    return interpolated_tensor

def normalize_tensor(input_tensor):
    """
    Normalizes the input tensor to the range [0, 1].
    
    Args:
        input_tensor (torch.Tensor): The input one-dimensional tensor.
        
    Returns:
        torch.Tensor: The normalized tensor.
    """
    min_val = torch.min(input_tensor)
    max_val = torch.max(input_tensor)
    
    normalized_tensor = (input_tensor - min_val) / (max_val - min_val)
    
    return normalized_tensor

def second_derivative(tensor, step_size):
    # Calculate the first derivative using finite differences
    first_derivative = (tensor[2:] - 2 * tensor[1:-1] + tensor[:-2]) / (step_size ** 2)
    return first_derivative

class Process(object):
    def __init__(self, num):
        self.num = num
    
    def prepro(self, length, piece):
        sig = np.load(path + '{0:0=d}'.format(self.num) + '.npy')
        sim = np.mean(sig, axis=1)
        self.filtered = hp.filter_signal(sim, [0.7, 3.5], sample_rate=30.0, order=3, filtertype='bandpass')
        wd, self.m = hp.process(self.filtered, sample_rate = 30.0, high_precision=True, clean_rr=True)
        masks = wd['RR_masklist']
        self.peak_list = wd['peaklist']
        self.peak_list = list(map(int, self.peak_list))
        ind = self.auth_index(masks, piece)
        sub_al = []
        start = self.peak_list[ind]
        end = self.peak_list[ind + piece]
        sub = self.filtered[start : end]
        original_sig = torch.tensor(normalization(sub))
        interval = int(len(original_sig) / piece)
        points_n = []
        for i in range(piece):
            points_n.append(i * interval + torch.argmin(original_sig[i * interval:(i + 1) * interval,]))
        pulses_n = []
        orig_n = []
        for i in range(len(points_n) - 1):
            pulses_n.append(original_sig[points_n[i]: points_n[i + 1]])
            orig_n.append(interpolate_tensor(normalize_tensor(pulses_n[-1]), 256))
            sub_al.append(interpolate_tensor(normalize_tensor(second_derivative(pulses_n[-1], 1/30.0)), 256))
        return orig_n, sub_al
    
    def auth_index(self, ma, step):
        for i in range(len(ma)):
            if ma[i] == 0:
                for j in range(step):
                    if j == step - 1 and ma[i + j] == 0:
                        return i
                    elif ma[i + j] == 0:
                        continue
                    else:
                        break