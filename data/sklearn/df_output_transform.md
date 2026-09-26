<!-- Converted from https://github.com/scikit-learn/scikit-learn/blob/1.9.1/doc/modules/df_output_transform.rst (BSD-3-Clause) -->

# Pandas/Polars Output for Transformers with `set_output` API

This part of the user guide explains how scikit-learn supports tabular data.

## Propagation of Feature Names

By default, scikit-learn transformers (estimators with a `transform`
method) return numpy arrays (sometimes also sparse arrays). Because numpy arrays do
not provide names for the indices of axes/dimensions, prior to version 1.0
the `pipeline.Pipeline` did not know how to propagate feature names:

- The single step estimators did not know how to handle incoming feature names.
- The pipeline did not know how to pass feature names from step to step.

In practice, a lot of use cases start with tabular data like a pandas dataframe or a
polars dataframe
which have column/feature names.

A first step to support this important use case was made by the addition of the
`compose.ColumnTransformer` in version 0.20.
It acts as a gateway to apply different estimators on the different features. Most
notably it understands incoming feature names.

It was then properly solved by SLEP007: Feature names, their generation and the API
and fully implemented in version 1.1, see
the release highlights 1.0 and
release highlights 1.1.
When an estimator is passed a dataframe during fit, the estimator will
set a `feature_names_in_` attribute containing the feature names. It understands pandas
dataframes as well as dataframes with the Python dataframe interchange protocol `__dataframe__`.
Furthermore, fitted estimators have the method `get_feature_names_out`. The
`get_feature_names_out` of a transformer returns⸺you guessed it⸺the feature names of
what `transform` returns.

## Introducing the `set_output` API

A further major step to support dataframes in a "dataframe in, dataframe out" fashion was
SLEP018,
implemented for pandas dataframes in version 1.2 and for
polars dataframes in version 1.4. It introduced the
`set_output` API to configure transformers to output pandas or polars DataFrames.
The output of transformers can be configured per estimator by calling
the `set_output` method or globally, by setting `set_config(transform_output="pandas")`.
Set it to `"polars"` instead of `"pandas"` if you want the same thing to happen but with
polars DataFrames.

The usage is basically as follows:

```
>>> import numpy as np
>>> import pandas as pd
>>> from sklearn.compose import ColumnTransformer
>>> from sklearn.pipeline import make_pipeline
>>> from sklearn.preprocessing import OneHotEncoder
>>> from sklearn.linear_model import LinearRegression

>>> X = pd.DataFrame(
...     {"animals": ["cat", "cat", "dog", "dog"], "numeric": np.linspace(-1, 1, 4)}
... )
>>> y = np.array([-1.5, 0, 0.1, 1.0])
>>> ct = ColumnTransformer(
...     [("categorical", OneHotEncoder(sparse_output=False), ["animals"])],
...     remainder="passthrough",
... )
>>> model = make_pipeline(ct, LinearRegression()).fit(X, y)
>>> model.feature_names_in_
array(['animals', 'numeric'], dtype=object)
>>> model[0].get_feature_names_out()
array(['categorical__animals_cat', 'categorical__animals_dog',
   'remainder__numeric'], dtype=object)
>>> model[0].transform(X)
array([[ 1.        ,  0.        , -1.        ],
       [ 1.        ,  0.        , -0.33333333],
       [ 0.        ,  1.        ,  0.33333333],
       [ 0.        ,  1.        ,  1.        ]])
```

Now the same, but with pandas set as output:

```
>>> from sklearn import set_config
>>> set_config(transform_output="pandas")
>>> model[0].transform(X)
       c...
```

To return to the default, simply run:

```
>>> set_config(transform_output="default")
```

A more detailed example can be found in
sphx glr auto examples miscellaneous plot set output.py.
