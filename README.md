# RandomForestRegressorModel
Project Description
This project uses a Random Forest Regressor model to predict stock levels based on sales data and temperature data. The project includes code to load data, split the data into predictor and target variables, train the model using cross-validation, and identify significant features.

Code Description
The code is written in Python and uses several libraries, including Pandas, Scikit-learn, Matplotlib, and NumPy.

The main script is predict_stock_levels.py, which contains the following functions:

load_data(sales_path: str, stock_path: str, temp_path: str): Loads three separate CSV files into three Pandas DataFrames.

create_target_and_predictors(data: pd.DataFrame = None, target: str = "estimated_stock_pct"): Separates the columns into two parts, the target column and a set of predictor variables (X and y).

train_algorithm_with_cross_validation(X: pd.DataFrame = None, y: pd.Series = None, K: int = 10, SPLIT: float = 0.8): Trains a Random Forest Regressor model across K folds using cross-validation.

features = [i.split("__")[0] for i in X.columns]: Identifies significant features.

The script also includes several import statements and an example of how to use the functions.

Instructions
To use this code, follow these steps:

Clone the repository from GitHub.
Install the required libraries by running pip install -r requirements.txt.
Run the predict_stock_levels.py script.
Follow the example to load data, split the data into predictor and target variables, train the model using cross-validation, and identify significant features.
References
Pandas documentation: https://pandas.pydata.org/docs/
Scikit-learn documentation: https://scikit-learn.org/stable/documentation.html
Matplotlib documentation: https://matplotlib.org/stable/contents.html
NumPy documentation: https://numpy.org/doc/stable/
