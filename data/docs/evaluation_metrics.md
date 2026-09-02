# Evaluation Metrics for Classification

Accuracy — the fraction of correct predictions — is the most common metric, but
on its own it can be dangerously misleading.

## Why accuracy alone is not enough

On an imbalanced dataset, a model that always predicts the majority class can
have high accuracy while being useless. A fraud detector on data that is 99%
legitimate reaches 99% accuracy by never flagging fraud. The per-class picture
matters.

## Precision, recall, and F1

- **Precision** = of the items predicted positive, how many really are. High
  precision means few false alarms.
- **Recall** = of the truly positive items, how many were found. High recall
  means few misses.
- **F1** is the harmonic mean of precision and recall, rewarding a balance.
- **Macro-F1** averages the per-class F1 scores equally, so a collapse on a
  minority class shows up even when overall accuracy looks fine.

## Other useful tools

A **confusion matrix** shows exactly which classes are mistaken for which.
**ROC-AUC** measures ranking quality across thresholds. For regression, report
error in the units people care about — RMSE or MAE — alongside R².
