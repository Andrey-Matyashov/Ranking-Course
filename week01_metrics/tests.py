import pytest
from week01_metrics.metrics import (
    compute_gain,
    dcg,
    ndcg,
    precission_at_k,
    reciprocal_rank,
    p_found,
    num_swapped_pairs
)
import torch
import math

def test_compute_gain():
    # Test with default gain scheme
    assert compute_gain(1.0, 'const') == 1.0
    # Test with logarithmic gain scheme
    assert math.isclose(compute_gain(1.0, 'log2'), math.log2(2))

def test_dcg():
    ys_true = torch.tensor([1, 0, 1])
    ys_pred = torch.tensor([1, 1, 0])
    assert math.isclose(dcg(ys_true, ys_pred, 'const'), 1.0)
    assert math.isclose(dcg(ys_true, ys_pred, 'log2'), math.log2(2) + math.log2(4))

def test_ndcg():
    ys_true = torch.tensor([1, 0, 1])
    ys_pred = torch.tensor([1, 1, 0])
    assert math.isclose(ndcg(ys_true, ys_pred, 'const'), 1.0)
    assert math.isclose(ndcg(ys_true, ys_pred, 'log2'), (math.log2(2) + math.log2(4)) / (math.log2(2) + math.log2(3)))

def test_precission_at_k():
    ys_true = torch.tensor([1, 0, 1])
    ys_pred = torch.tensor([1, 1, 0])
    assert precission_at_k(ys_true, ys_pred, 1) == 1.0
    assert precission_at_k(ys_true, ys_pred, 2) == 0.5

def test_reciprocal_rank():
    ys_true = torch.tensor([1, 0, 1])
    ys_pred = torch.tensor([1, 1, 0])
    assert reciprocal_rank(ys_true, ys_pred) == 1.0

def test_p_found():
    ys_true = torch.tensor([1, 0, 1])
    ys_pred = torch.tensor([1, 1, 0])
    assert p_found(ys_true, ys_pred) == 1.0

def test_num_swapped_pairs():
    ys_true = torch.tensor([1, 0, 1])
    ys_pred = torch.tensor([1, 1, 0])
    assert num_swapped_pairs(ys_true, ys_pred) == 1