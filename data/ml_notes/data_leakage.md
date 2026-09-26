# Data Leakage in Machine Learning

Data leakage happens when information that would not be available at prediction
time leaks into the training process, producing models that look excellent in
evaluation but fail in production.

## Common forms

- **Preprocessing leakage.** Fitting a scaler, imputer, or feature selector on
  the whole dataset before splitting lets test-set statistics influence
  training. The fix is to fit every transform on the training fold only, which
  is why scikit-learn `Pipeline` objects are the safe default.
- **Target leakage.** Including a feature that is a proxy for the label, or that
  is only known after the outcome, inflates accuracy. Example: using "number of
  late-payment reminders" to predict default.
- **Duplicate / near-duplicate leakage.** When augmented or near-identical
  copies of the same record appear in both train and test, the model can score
  highly by memorising specific items rather than learning a general pattern.

## Detecting it

A result that is far better than the problem should allow is the first warning
sign. Useful controls include a label-permutation test (shuffle the labels and
retrain; a clean pipeline drops to chance), a nearest-neighbour audit for
near-duplicates across the split, and always comparing accuracy against the
chance baseline for the number of classes.
