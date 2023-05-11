import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler


import matplotlib.pyplot as plt
import numpy as np


# Load data

def load_data(sales_path: str, stock_path: str, temp_path: str):
    """
    This function loads three separate CSV files into three Pandas DataFrames. 
    The function takes in three path strings representing the relative path of each CSV file. 
    The function returns three separate Pandas DataFrames 
    representing the sales data, stock levels data, and temperature data.

    """

    sales_df = pd.read_csv("/content/drive/MyDrive/CVS/sales.csv")
    sales_df.drop(columns=["Unnamed: 0"], inplace=True, errors='ignore')

    stock_df = pd.read_csv("/content/drive/MyDrive/CVS/sensor_stock_levels.csv")
    stock_df.drop(columns=["Unnamed: 0"], inplace=True, errors='ignore')

    temp_df = pd.read_csv("/content/drive/MyDrive/CVS/sensor_storage_temperature.csv")
    temp_df.drop(columns=["Unnamed: 0"], inplace=True, errors='ignore')

    return sales_df, stock_df, temp_df


# Create target variable and predictor variables
def create_target_and_predictors(
    data: pd.DataFrame = None, 
    target: str = "estimated_stock_pct"
):
    """
    This function receives a Pandas DataFrame as input and separates the columns into two parts, 
    the target column and a set of predictor variables (X and y). 
    This split of the data will be used to train a supervised machine learning model. 
    The target variable is optional and is used to specify which column in the data should be considered the target. 
    The function returns two objects: 
    X, which is a Pandas DataFrame of predictor variables, 
    and y, which is a Pandas Series of the target variable.
    """

    # Check to see if the target variable is present in the data
    if target not in data.columns:
        raise Exception(f"Target: {target} is not present in the data")
    
    X = data.drop(columns=[target])
    y = data[target]
    return X, y

# Train algorithm
def train_algorithm_with_cross_validation(
    X: pd.DataFrame = None, 
    y: pd.Series = None
    K: int = 10,
    SPLIT: float = 0.8
):
    """
    This function trains a Random Forest Regressor model across K folds using cross-validation and returns no output. 
    The predictor variables are passed as a Pandas DataFrame X, and the target variable is passed as a Pandas Series y.
    """

    # Create a list that will store the accuracies of each fold
    accuracy = []

    # Enter a loop to run K folds of cross-validation
    for fold in range(0, K):

        # Instantiate algorithm and scaler
        model = RandomForestRegressor()
        scaler = StandardScaler()

        # Create training and test samples
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=SPLIT, random_state=42)

        # Scale X data, we scale the data because it helps the algorithm to converge
        # and helps the algorithm to not be greedy with large values
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        # Train model
        trained_model = model.fit(X_train, y_train)

        # Generate predictions on test sample
        y_pred = trained_model.predict(X_test)

        # Compute accuracy, using mean absolute error
        mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
        accuracy.append(mae)
        print(f"Fold {fold + 1}: MAE = {mae:.3f}")

    # Finish by computing the average MAE across all folds
    print(f"Average MAE: {(sum(accuracy) / len(accuracy)):.2f}")



#Model can identify significant features

features = [i.split("__")[0] for i in X.columns]
importances = model.feature_importances_
indices = np.argsort(importances)

fig, ax = plt.subplots(figsize=(10, 20))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()