<!-- Converted from https://github.com/scikit-learn/scikit-learn/blob/1.9.1/doc/modules/unsupervised_reduction.rst (BSD-3-Clause) -->

# Unsupervised dimensionality reduction

If your number of features is high, it may be useful to reduce it with an
unsupervised step prior to supervised steps. Many of the
unsupervised-learning methods implement a `transform` method that
can be used to reduce the dimensionality. Below we discuss two specific
examples of this pattern that are heavily used.

****Pipelining****
The unsupervised data reduction and the supervised estimator can be
chained in one step. See pipeline.

## PCA: principal component analysis

`decomposition.PCA` looks for a combination of features that
capture well the variance of the original features. See decompositions.

## Random projections

The module: `random_projection` provides several tools for data
reduction by random projections. See the relevant section of the
documentation: random projection.

## Feature agglomeration

`cluster.FeatureAgglomeration` applies
hierarchical clustering to group together features that behave
similarly.

****Feature scaling****
Note that if features have very different scaling or statistical
properties, `cluster.FeatureAgglomeration` may not be able to
capture the links between related features. Using a
`preprocessing.StandardScaler` can be useful in these settings.
