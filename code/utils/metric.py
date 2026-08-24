"""
Biomass Estimation Evaluation Metrics
=====================================

This module demonstrates three common evaluation metrics used to assess 
the accuracy of predicted Length, Width, and Weight against ground-truth values.

Metrics:
--------
1. MD   (Mean Difference)          → Bias of the predictions
2. MAD  (Mean Absolute Difference) → Average magnitude of errors
3. MPAE (Mean Percentage Absolute Error) → Relative error in percentage

Formulas:
---------
Given predictions ŷ and ground-truth value y:

    MD   = mean(ŷ) - y
    MAD  = mean( |ŷ - y| )
    MPAE = mean( |ŷ - y| / y ) × 100
"""

import numpy as np


def compute_metrics(predictions, ground_truth):
    """
    Compute MD, MAD and MPAE.
    
    Parameters
    ----------
    predictions : array-like
        List or array of predicted values
    ground_truth : float
        Single ground-truth reference value
        
    Returns
    -------
    md   : float
        Mean Difference (Bias)
    mad  : float
        Mean Absolute Difference
    mpae : float
        Mean Percentage Absolute Error (%)
    """
    pred = np.asarray(predictions, dtype=float)
    pred = pred[np.isfinite(pred)]          # remove NaN / Inf
    
    if len(pred) == 0:
        return np.nan, np.nan, np.nan
    
    md   = np.mean(pred) - ground_truth
    mad  = np.mean(np.abs(pred - ground_truth))
    mpae = np.mean(np.abs(pred - ground_truth) / ground_truth) * 100.0
    
    return md, mad, mpae


# =========================================================
# Example 1 - Length estimation
# =========================================================
print("=" * 55)
print("Example 1: Length Estimation")
print("=" * 55)

gt_length = 18.50          # Ground-truth average length (cm)
pred_lengths = [17.8, 19.1, 18.3, 17.2, 19.6, 18.0, 18.9]

md, mad, mpae = compute_metrics(pred_lengths, gt_length)

print(f"Ground Truth Length : {gt_length:.2f} cm")
print(f"Predictions         : {pred_lengths}")
print(f"MD   (Bias)         : {md:+.3f} cm")
print(f"MAD                 : {mad:.3f} cm")
print(f"MPAE                : {mpae:.2f} %")


# =========================================================
# Example 2 - Width estimation
# =========================================================
print("\n" + "=" * 55)
print("Example 2: Width Estimation")
print("=" * 55)

gt_width = 5.80            # Ground-truth average width (cm)
pred_widths = [5.5, 6.1, 5.7, 5.3, 6.0, 5.9, 5.6]

md, mad, mpae = compute_metrics(pred_widths, gt_width)

print(f"Ground Truth Width  : {gt_width:.2f} cm")
print(f"Predictions         : {pred_widths}")
print(f"MD   (Bias)         : {md:+.3f} cm")
print(f"MAD                 : {mad:.3f} cm")
print(f"MPAE                : {mpae:.2f} %")


# =========================================================
# Example 3 - Weight estimation
# =========================================================
print("\n" + "=" * 55)
print("Example 3: Weight Estimation")
print("=" * 55)

gt_weight = 125.0          # Ground-truth average weight (g)
pred_weights = [118.5, 132.0, 121.3, 115.8, 129.7, 123.4, 127.1]

md, mad, mpae = compute_metrics(pred_weights, gt_weight)

print(f"Ground Truth Weight : {gt_weight:.1f} g")
print(f"Predictions         : {pred_weights}")
print(f"MD   (Bias)         : {md:+.2f} g")
print(f"MAD                 : {mad:.2f} g")
print(f"MPAE                : {mpae:.2f} %")


# =========================================================
# Interpretation guide
# =========================================================
print("\n" + "=" * 55)
print("How to interpret the metrics")
print("=" * 55)
print("""
MD (Mean Difference):
    - Positive value → overestimation
    - Negative value → underestimation
    - Closer to zero is better

MAD (Mean Absolute Difference):
    - Average size of the error (ignores direction)
    - Smaller is better

MPAE (Mean Percentage Absolute Error):
    - Relative error in percentage
    - Useful for comparing different scales (cm vs g)
    - Smaller is better
""")
