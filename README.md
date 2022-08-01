# feature_analysis

feature_analysis is a Python library for evaluating features to be used in a model.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash
pip install -i https://test.pypi.org/simple/ feature-analysis-karsh==0.0.10
```

## Usage

### Evaluating Models in Binary Classification
1) Import pandas
2) Import binary_class from feature_analysis
3) Load dataset

```python
import pandas as pd
from feature_analysis import binary_class

df = pd.read_csv("test.csv")
```

4) Create a feature_analysis object using inputs dataframe and target
- Note it is important to specify what your "bad class" is
- Default for bad_class is 1

```python
report = binary_class.feature_analysis(df, "Target", bad_class="Yes")
```

5) Can use feature_analysis to find missing_rate:
```python
report.missing_rate()
```
<img width="232" alt="image" src="https://user-images.githubusercontent.com/62733489/176062514-03dadef0-2e7c-4b08-9798-a4447c690d2a.png">

6) Can use feature_analysis to find zero_rate:
```python
report.zero_rate()
```
<img width="203" alt="image" src="https://user-images.githubusercontent.com/62733489/176062685-224804bc-6f30-408d-a1f3-5b1bca8bad94.png">

7) Can use feature_analysis to get Missing, Zero, AUC, KS, IV using analyse:
```python
report.analyse()
```
<img width="387" alt="image" src="https://user-images.githubusercontent.com/62733489/176064028-3ec5b076-b216-468f-a8a3-0655f6a4146e.png">

8) Can see how IV for each feature was calculated via iv_scores:
```python
report.iv_scores()
```
<img width="342" alt="image" src="https://user-images.githubusercontent.com/62733489/176064174-ad572539-03d3-44ba-966d-95a43c15aeff.png">

9) feature_analysis can also help you check how much your features vary over time using PSI
- Requires a column which contains datetime
- Works best if columns is already binned (with as few unique values as possible)
- If datetime column has many different dates, ensure that column is in datetime data type to allow function to bin
- Need to add one more date column input when creating feature_analysis object:
```python
report = binary_class.feature_analysis(df, "Target", date="Date", bad_class="Yes")
```
- Previous methods (missing_rate, zero_rate, analysis, iv_scores) all remain same

10) Check variance in time for each feature using stability (calculates PSI):
- bin_dates set to True IF require binning for date column (MUST be numerical or datetime data type) 
- If already binned or restrained to small number of unique date values set bin_dates to false
```python
report.stability(True)
```
<img width="607" alt="image" src="https://user-images.githubusercontent.com/62733489/176066085-f02c3eca-887a-460c-8d3b-332b52ff6604.png">

11) Can download an excel file with all the different metrics using to_excel
- Standard filename is report.xlsx, can change using name
```python
report.stability(name="report2.xlsx")
```

## Note
1) Missing Values
- Missing values in Binary Classification are default dealt in the following way:
- Constants: "NA"
- Numerical: -999999999  
  
- Missing values in Regression are default dealt in the following way:
- Constants: "NA"
- Numerical: Mean value of Column

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)