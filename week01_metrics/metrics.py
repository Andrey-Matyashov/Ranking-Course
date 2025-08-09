import math
import torch

from math import log2

from torch import Tensor, sort


def num_swapped_pairs(ys_true: Tensor, ys_pred: Tensor) -> int:
    ys_pred_sorted, indicies = sort(ys_pred)
    ys_true_ordered = ys_true[indicies]
    cnt_swapped = 0
    for i in range(len(ys_true_ordered) - 1):
        for j in range(i + 1, len(ys_true_ordered)):
            if ys_true_ordered[i] > ys_true_ordered[j]:
                cnt_swapped += 1
    return cnt_swapped

def compute_gain(y_value: float, gain_scheme: str) -> float:
    if gain_scheme == "exp2":
        gain = 2 ** y_value - 1
    elif gain_scheme == "const":
        gain = y_value
    else:
        raise ValueError(f"{gain_scheme} method not supported, only exp2 and const.")
    return float(gain)


def dcg(ys_true: Tensor, ys_pred: Tensor, gain_scheme: str) -> float:
    _, argsort = torch.sort(ys_pred, descending=True, dim=0)
    ys_true_sorted = ys_true[argsort]
    ret = 0
    for idx, cur_y in enumerate(ys_true_sorted, 1):
        gain = compute_gain(cur_y, gain_scheme)
        ret += gain / math.log2(idx + 1)
    return ret


def ndcg(ys_true: Tensor, ys_pred: Tensor, gain_scheme: str = 'const') -> float:
    pred_dcg = dcg(ys_true, ys_pred, gain_scheme)
    ideal_dcg = dcg(ys_true, ys_true, gain_scheme)
    
    ndcg = pred_dcg / ideal_dcg
    return ndcg



def precission_at_k(ys_true: Tensor, ys_pred: Tensor, k: int) -> float:
    if ys_true.sum() == 0:
        return -1
    _, argsort = torch.sort(ys_pred, descending=True, dim=0)
    ys_true_sorted = ys_true[argsort]
    hits = ys_true_sorted[:k].sum()
    prec = hits / min(ys_true.sum(), k)
    return float(prec)


def reciprocal_rank(ys_true: Tensor, ys_pred: Tensor) -> float:
    _, argsort = torch.sort(ys_pred, descending=True, dim=0)
    ys_true_sorted = ys_true[argsort]
    
    for idx, cur_y in enumerate(ys_true_sorted, 1):
        if cur_y == 1:
            return 1 / idx
    return 0

def p_found(ys_true: Tensor, ys_pred: Tensor, p_break: float = 0.15 ) -> float:
    p_look = 1
    p_found = 0
    _, argsort = torch.sort(ys_pred, descending=True, dim=0)
    ys_true_sorted = ys_true[argsort]

    for cur_y in ys_true_sorted:
        p_found += p_look * float(cur_y)
        p_look = p_look * (1 - float(cur_y)) * (1 - p_break)
    
    return p_found


def average_precision(ys_true: torch.Tensor, ys_pred: torch.Tensor) -> float:
    if ys_true.sum() == 0:
        return -1
    _, argsort = torch.sort(ys_pred, descending=True, dim=0)
    ys_true_sorted = ys_true[argsort]
    rolling_sum = 0
    num_correct_ans = 0
    
    for idx, cur_y in enumerate(ys_true_sorted, start=1):
        if cur_y == 1:
            num_correct_ans += 1
            rolling_sum += num_correct_ans / idx
    if num_correct_ans == 0:
        return 0
    else:
        return rolling_sum / num_correct_ans    


if __name__ == '__main__':
    
    a = torch.tensor([3, 2, 1])
    b = torch.tensor([1, 2, 3])
    print(num_swapped_pairs(a, b))