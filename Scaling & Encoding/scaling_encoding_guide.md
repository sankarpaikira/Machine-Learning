# Scaling and Encoding Guide

This guide is based on the scaling and encoding techniques already used in your notebooks.

## Related Files in This Project

- `Scaling & Encoding/5th_scaling.ipynb`
- `Scaling & Encoding/6th_encoding_data.ipynb`
- `Scaling & Encoding/7th_encoding (with sample data).ipynb`  
  Note: this file is currently empty on disk.
- `Scaling & Encoding/8th_encoding (with dataset).ipynb`
- `Scaling & Encoding/9th_scaling_encoding_task.ipynb`

## Datasets You Already Have

- `datasets/insurance.csv`
- `datasets/Student_Performance.csv`
- `datasets/loan_data.csv`
- `datasets/loan_approval_dataset.csv`
- `datasets/Housing.csv`

## Quick Rule

- Use scaling for numeric columns.
- Use encoding for categorical columns.
- Fit transformers on `x_train` only, then transform both `x_train` and `x_test`.

---

## 1. StandardScaler

### What it does

It changes each numeric feature so it is centered around `0` with standard deviation close to `1`.

### Best when

- numeric columns have different ranges
- your model is distance-based or gradient-based
- outliers are not too extreme

### Common models where it helps

- Linear Regression
- Logistic Regression
- KNN
- SVM
- PCA

### Avoid or be careful when

- data has many strong outliers
- you want values restricted to a fixed range like `0` to `1`

### Simple examples

1. `datasets/insurance.csv`
   Use for `age`, `bmi`, `children` when training linear or logistic models.
2. `datasets/loan_data.csv`
   Use for columns like `person_income`, `loan_amnt`, `credit_score`.
3. Wine dataset
   Good for chemical features like alcohol, acidity, sulphates before KNN or SVM.

### Example code

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
xtrain[num_cols] = scaler.fit_transform(xtrain[num_cols])
xtest[num_cols] = scaler.transform(xtest[num_cols])
```

### Parameters it accepts

`StandardScaler(copy=True, with_mean=True, with_std=True)`

- `copy=True`
  Creates a new scaled output instead of forcing in-place changes.
  Use default in most cases.
  Change it only when you want memory optimization and are sure in-place behavior is safe.
- `with_mean=True`
  Centers data by subtracting the mean.
  Use this in most normal tabular ML cases.
  Do not use it with sparse input matrices because centering sparse data is inefficient and can fail.
- `with_std=True`
  Scales data to unit standard deviation.
  Keep it `True` when you want proper standardization.
  Set it to `False` only if you want centering without variance scaling.

### When to change parameters

- use `with_mean=False` for sparse matrix input
- use `with_std=False` if only mean-centering is needed
- keep defaults for normal pandas or NumPy numeric data

---

## 2. MinMaxScaler

### What it does

It rescales values into a fixed range, usually `0` to `1`.

### Best when

- you want all values in the same fixed range
- data has no major outliers
- you are working with neural networks or distance-based models

### Common models where it helps

- KNN
- Neural Networks
- KMeans
- models using gradient descent

### Avoid or be careful when

- there are large outliers because they stretch the whole range

### Simple examples

1. `datasets/Student_Performance.csv`
   Good for columns like `Hours Studied` and `Previous Scores`.
2. `datasets/Housing.csv`
   Good if features like area and bedroom-related values are on different ranges and outliers are limited.
3. Image pixel data
   Very common because pixel values are often scaled from `0-255` to `0-1`.

### Example code

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
xtrain[num_cols] = scaler.fit_transform(xtrain[num_cols])
xtest[num_cols] = scaler.transform(xtest[num_cols])
```

### Parameters it accepts

`MinMaxScaler(feature_range=(0, 1), copy=True, clip=False)`

- `feature_range=(0, 1)`
  Sets the output range.
  Use `(0, 1)` in most cases.
  Use something like `(-1, 1)` when your model or activation functions work better around zero.
- `copy=True`
  Returns a copied transformed output.
  Keep default unless you intentionally want in-place style behavior.
- `clip=False`
  If test values go outside the training range, this controls whether transformed values are clipped into the selected range.
  Use `clip=True` when you want strict bounded output in production.
  Keep `False` when you want to preserve how far outside the training range new values are.

### When to change parameters

- use `feature_range=(-1, 1)` for centered scaled inputs
- use `clip=True` when unseen test values must stay inside the same range
- keep defaults for ordinary `0-1` scaling

---

## 3. RobustScaler

### What it does

It scales data using the median and IQR, so it is less affected by outliers.

### Best when

- numeric columns contain outliers
- you still want scaling but standard scaling gets distorted

### Common models where it helps

- Linear Regression
- Logistic Regression
- KNN
- SVM

### Avoid or be careful when

- data is already clean and you specifically want a `0-1` range

### Simple examples

1. `datasets/loan_data.csv`
   Income and loan amount often contain extreme values.
2. `datasets/insurance.csv`
   If `bmi` or `charges` related predictors look skewed or have outliers.
3. Sales or transaction datasets
   Useful when a few customers spend far more than the rest.

### Example code

```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
xtrain[num_cols] = scaler.fit_transform(xtrain[num_cols])
xtest[num_cols] = scaler.transform(xtest[num_cols])
```

### Parameters it accepts

`RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0), copy=True, unit_variance=False)`

- `with_centering=True`
  Centers using the median.
  Use default for regular dense data.
  Avoid centering for sparse matrices.
- `with_scaling=True`
  Scales using the interquartile range.
  Keep this `True` when you want robust scaling.
- `quantile_range=(25.0, 75.0)`
  Chooses which quantiles define the scaling range.
  Default uses IQR.
  Use a wider range like `(10, 90)` if you want scaling influenced by more of the data.
  Use a narrower range only when you want stronger robustness to extreme values.
- `copy=True`
  Returns a copied transformed output.
- `unit_variance=False`
  If `True`, scaling is adjusted so normally distributed features have variance close to `1`.
  Use this when you want something closer to standard-scaler-like variance behavior but still robust to outliers.

### When to change parameters

- use `quantile_range=(10, 90)` when default IQR feels too aggressive
- use `unit_variance=True` if downstream models benefit from features closer to equal variance
- use `with_centering=False` for sparse inputs

---

## 4. LabelEncoder

### What it does

It converts labels into numbers like:

- `yes -> 1`
- `no -> 0`

### Best when

- encoding the target column `y`
- the target has class names as text

### Important note

Use `LabelEncoder` mainly for the target column, not for multiple input feature columns together.

### Best scenarios

- binary classification targets
- multiclass target labels

### Avoid or be careful when

- encoding input features with no natural order because the model may think one category is bigger than another

### Simple examples

1. `datasets/loan_approval_dataset.csv`
   Encode target like `Loan_Approved`.
2. Iris dataset
   Encode target labels like `setosa`, `versicolor`, `virginica`.
3. Spam detection
   Encode target values like `spam` and `not_spam`.

### Example code

```python
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
ytrain = encoder.fit_transform(ytrain)
ytest = encoder.transform(ytest)
```

### Parameters it accepts

`LabelEncoder()` has no constructor parameters.

### Important behavior

- it sorts unique labels and assigns integer values
- it is best for `y`, not for many `X` columns
- unseen labels during `transform()` will raise an error

### When to use it

- target column has text labels
- you need numeric target classes for sklearn models

---

## 5. OrdinalEncoder

### What it does

It converts categories into ordered numbers like:

- `small -> 0`
- `medium -> 1`
- `large -> 2`

### Best when

- categories have a real order
- higher or lower category has actual meaning

### Good examples of ordered categories

- education level
- t-shirt size
- satisfaction rating
- risk level

### Avoid or be careful when

- categories are just names with no order, such as `red`, `blue`, `green`

### Simple examples

1. `datasets/loan_data.csv`
   `person_education` can be encoded only if you define a meaningful order.
2. A shirt-size dataset
   `S < M < L < XL`
3. Feedback dataset
   `poor < average < good < excellent`

### Example code

```python
from sklearn.preprocessing import OrdinalEncoder

encoder = OrdinalEncoder(
    categories=[['high school', 'bachelor', 'master', 'phd']],
    handle_unknown='use_encoded_value',
    unknown_value=-1
)

xtrain[['person_education']] = encoder.fit_transform(xtrain[['person_education']])
xtest[['person_education']] = encoder.transform(xtest[['person_education']])
```

### Parameters it accepts

`OrdinalEncoder(categories='auto', dtype=<numeric type>, handle_unknown='error', unknown_value=None, encoded_missing_value=np.nan, min_frequency=None, max_categories=None)`

- `categories='auto'`
  Learns categories from the training data automatically.
  Use this when order does not need to be manually controlled.
  If the order matters, pass your own ordered list.
- `dtype`
  Controls output type, usually float.
  Change it when you want integer output and your missing/unknown handling allows it.
- `handle_unknown='error'`
  Default raises an error if test data contains a new category.
  Use `handle_unknown='use_encoded_value'` when unseen categories may appear.
- `unknown_value=None`
  Required when `handle_unknown='use_encoded_value'`.
  Common choices are `-1` or another reserved code.
- `encoded_missing_value=np.nan`
  Decides how missing values are encoded.
  Keep as `np.nan` if you want missingness preserved.
  Change it when the next step cannot handle NaNs.
- `min_frequency=None`
  Groups rare categories together when set.
  Use this when some categories are too rare.
- `max_categories=None`
  Limits the number of output category codes by grouping infrequent levels.
  Use this when you want to reduce category complexity.

### When to change parameters

- pass manual `categories=[...]` when true order matters
- use `handle_unknown='use_encoded_value'` and `unknown_value=-1` for safer production inference
- use `encoded_missing_value` when missing values need a fixed code
- use `min_frequency` or `max_categories` when categories are too fragmented

---

## 6. OneHotEncoder

### What it does

It creates one new column for each category.

Example:

- `region_north`
- `region_south`
- `region_east`
- `region_west`

### Best when

- categories have no natural order
- there are few or moderate unique values

### Best scenarios

- color
- city group
- gender
- smoker or non-smoker
- loan purpose

### Avoid or be careful when

- a column has too many unique values, because it creates too many new columns

### Simple examples

1. `datasets/insurance.csv`
   `sex`, `smoker`, `region`
2. `datasets/Student_Performance.csv`
   `Extracurricular Activities`
3. `datasets/loan_data.csv`
   `loan_intent`, `person_home_ownership`, `person_gender`

### Why `sparse_output=False` is used

By default, `OneHotEncoder` returns a sparse matrix because most values are `0`.

Use `sparse_output=False` when:

- you want a normal NumPy array
- you want to add encoded values back into a pandas DataFrame easily
- you want easier printing and debugging

If you do not use it:

- output stays as a sparse matrix
- this is more memory-efficient
- but it is less convenient for direct DataFrame assignment

### Example code

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

train_encoded = encoder.fit_transform(xtrain[obj_cols])
test_encoded = encoder.transform(xtest[obj_cols])

encoded_cols = encoder.get_feature_names_out(obj_cols)
```

### Parameters it accepts

`OneHotEncoder(categories='auto', drop=None, sparse_output=True, dtype=float, handle_unknown='error', min_frequency=None, max_categories=None, feature_name_combiner='concat')`

- `categories='auto'`
  Learns categories from training data.
  Pass your own categories only when you need fixed category order or strict schema control.
- `drop=None`
  Keeps all dummy columns.
  Use `drop='first'` when you want to remove one dummy per feature, often for linear models to reduce redundancy.
  Use `drop='if_binary'` for binary columns only.
- `sparse_output=True`
  Default returns sparse matrix output.
  Use `False` when you want a normal dense array for easy DataFrame handling.
- `dtype=float`
  Output data type.
  Often left as default.
  Change to `int` only if you specifically want integer dummy columns.
- `handle_unknown='error'`
  Raises an error for unseen categories.
  Use `handle_unknown='ignore'` in most real projects so new categories in test or production do not break the pipeline.
- `min_frequency=None`
  Groups rare categories into an infrequent bucket.
  Useful when many categories appear only a few times.
- `max_categories=None`
  Caps the number of categories kept per feature.
  Useful when cardinality is moderately high.
- `feature_name_combiner='concat'`
  Controls how output column names are built.
  Default is fine in almost all cases.

### When to change parameters

- use `sparse_output=False` when assigning back to pandas DataFrames
- use `handle_unknown='ignore'` for safer inference
- use `drop='first'` for linear-model dummy variable reduction
- use `min_frequency` or `max_categories` when categories are too many

---

## 7. TargetEncoder

### What it does

It replaces each category with a value derived from the target.

For example, if people from one category usually have higher target values, that category gets a higher encoded number.

### Best when

- a categorical feature has many unique values
- one-hot encoding would create too many columns
- you want to capture the relationship between category and target

### Common use cases

- zip code
- product ID
- neighborhood
- merchant ID
- job title with many unique values

### Avoid or be careful when

- the dataset is very small
- you do not control leakage correctly
- the encoded values are learned directly from the target, so train/test separation matters a lot

### Simple examples

1. High-cardinality location data
   Such as `zipcode -> house_price`.
2. E-commerce data
   `product_id -> sales` or `merchant_id -> fraud`.
3. Large loan or customer datasets
   Useful when a categorical column has many categories and one-hot encoding becomes too wide.

### Example code

```python
from category_encoders import TargetEncoder

encoder = TargetEncoder(cols=['high_card_col'], handle_unknown='value')

xtrain = encoder.fit_transform(xtrain, ytrain)
xtest = encoder.transform(xtest)
```

### Parameters it accepts

`TargetEncoder(verbose=0, cols=None, drop_invariant=False, return_df=True, handle_missing='value', handle_unknown='value', min_samples_leaf=20, smoothing=10, hierarchy=None)`

- `verbose=0`
  Controls logging.
  Keep `0` normally.
- `cols=None`
  Which columns to target-encode.
  If `None`, string columns are encoded automatically.
  In practice, it is better to pass the exact columns you want.
- `drop_invariant=False`
  Drops encoded columns that become constant.
  Use `True` when you want automatic cleanup of no-variance features.
- `return_df=True`
  Returns a pandas DataFrame instead of a NumPy array.
  Keep `True` if you are working in pandas notebooks.
- `handle_missing='value'`
  Missing categories are replaced with the global target mean by default.
  Use `'error'` to fail fast or `'return_nan'` if you want NaNs preserved.
- `handle_unknown='value'`
  Unseen categories are replaced with the global target mean by default.
  This is usually a safe practical choice.
- `min_samples_leaf=20`
  Regularization control based on category size.
  Lower values trust small categories more.
  Higher values shrink small categories more strongly toward the global mean.
- `smoothing=10`
  Controls how strongly category means are blended with the overall mean.
  Higher values mean stronger smoothing.
- `hierarchy=None`
  Lets you define grouped or hierarchical category relationships.
  Use only in advanced cases.

### When to change parameters

- set `cols=[...]` explicitly for clarity
- increase `smoothing` when you have noisy small categories
- increase `min_samples_leaf` when rare categories should be trusted less
- keep `handle_unknown='value'` in most production-style cases
- keep `return_df=True` if you want easy pandas integration

### Important note

Always fit target encoding on training data only.  
Never fit it on the full dataset before splitting.

---

## Which One Should I Use?

### For numeric columns

- use `StandardScaler` when ranges differ and outliers are limited
- use `MinMaxScaler` when you want values between `0` and `1`
- use `RobustScaler` when outliers are present

### For categorical columns

- use `LabelEncoder` for target labels
- use `OrdinalEncoder` for categories with real order
- use `OneHotEncoder` for categories with no order
- use `TargetEncoder` for high-cardinality categorical columns

---

## Very Practical Selection Table

| Data type | Situation | Best choice |
|---|---|---|
| Numeric | Different ranges, few outliers | `StandardScaler` |
| Numeric | Want fixed `0-1` range | `MinMaxScaler` |
| Numeric | Many outliers | `RobustScaler` |
| Categorical | Target labels | `LabelEncoder` |
| Categorical | Ordered categories | `OrdinalEncoder` |
| Categorical | Unordered categories, low cardinality | `OneHotEncoder` |
| Categorical | Unordered categories, high cardinality | `TargetEncoder` |

---

## Suggested Practice on Your Files

1. `datasets/insurance.csv`
   Use `StandardScaler` or `RobustScaler` for numeric columns and `OneHotEncoder` for `sex`, `smoker`, `region`.
2. `datasets/Student_Performance.csv`
   Use `MinMaxScaler` for numeric columns and `OneHotEncoder` for `Extracurricular Activities`.
3. `datasets/loan_data.csv`
   Use `RobustScaler` or `StandardScaler` for numeric columns, `OneHotEncoder` for unordered categoricals, and `OrdinalEncoder` only if you define a true order for education levels.
4. `datasets/loan_approval_dataset.csv`
   Use scaling on `Age`, `Salary`, `Credit_Score`, `Loan_Amount`, and encode categorical columns depending on whether they are ordered or unordered.

---

## Internet Sources Used

- scikit-learn `StandardScaler`: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
- scikit-learn `MinMaxScaler`: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
- scikit-learn `RobustScaler`: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html
- scikit-learn `OneHotEncoder`: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
- scikit-learn `OrdinalEncoder`: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html
- scikit-learn `LabelEncoder`: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
- Category Encoders `TargetEncoder`: https://contrib.scikit-learn.org/category_encoders/targetencoder.html
- UCI Adult dataset: https://archive.ics.uci.edu/dataset/2/adult
- UCI Wine Quality dataset: https://archive.ics.uci.edu/dataset/186/wine+quality
