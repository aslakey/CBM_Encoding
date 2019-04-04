import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import category_encoders as ce

import pandas as pd
from gaussian_inverse_gamma_encoder import GIGEncoder
from utils import *
import csv


def run_i_experiments():
    print("Loading Data")
    df = load_data()
    #columns:
    continuous = ['age','bmi']
    categorical = ['sex','children','smoker','region']

    X = df[continuous+categorical]
    y = df[['charges']]

    u_0 = np.mean(y)[0]
    v = np.std(y)[0]

    models = [
        Ridge(),
        RandomForestRegressor(n_estimators=100),
        GradientBoostingRegressor(),
        MLPRegressor()]
    #models = [RandomForestRegressor()]

    results = [['model','Encoder','R2','STD','Training Time','Sparsity','Dimensions']]

    for model in models:
        print("")
        print("----------------------")
        print("Testing Algorithm: ")
        print(type(model))
        print("----------------------")

        #TargetEncoder
        print("TargetEncoder Results:")
        r2, std, time, sparsity, dimensions = cv_regression(model, X, y, continuous, categorical, encoder=ce.TargetEncoder(return_df=False))
        results.append([type(model), 'TargetEncoder', r2, std, time, sparsity, dimensions])

        #OrdinalEncoder
        print("OrdinalEncoder Results:")
        r2, std, time, sparsity, dimensions = cv_regression(model, X, y, continuous, categorical, encoder=ce.OrdinalEncoder(return_df=False))
        results.append([type(model), 'OrdinalEncoder', r2, std, time, sparsity, dimensions])

        #BinaryEncoder
        print("BinaryEncoder Results:")
        r2, std, time, sparsity, dimensions = cv_regression(model, X, y, continuous, categorical, encoder=ce.BinaryEncoder(return_df=False))
        results.append([type(model), 'BinaryEncoder', r2, std, time, sparsity, dimensions])

        #HashingEncoder
        print("HashingEncoder Results:")
        r2, std, time, sparsity, dimensions = cv_regression(model, X, y, continuous, categorical, encoder=ce.HashingEncoder(return_df=False))
        results.append([type(model), 'HashingEncoder', r2, std, time, sparsity, dimensions])

        #OneHotEncoder
        print("OneHotEncoder Results:")
        r2, std, time, sparsity, dimensions = cv_regression(model, X, y, continuous, categorical, encoder=OneHotEncoder(handle_unknown='ignore', sparse=False))
        results.append([type(model), 'OneHotEncoder', r2, std, time, sparsity, dimensions])

        
        print("GIG Encoder (mean) Results:")
        r2, std, time, sparsity, dimensions = cv_regression(model, X, y, continuous, categorical, encoder=GIGEncoder(u_0=u_0, v=v))
        results.append([type(model), 'GIGEncoder (m)', r2, std, time, sparsity, dimensions])

        print("GIG Encoder (mean and variance) Results:")
        r2, std, time, sparsity, dimensions = cv_regression(model, X, y, continuous, categorical, encoder=GIGEncoder(u_0=u_0, v=v), moments='mv')
        results.append([type(model), 'GIGEncoder (mv)', r2, std, time, sparsity, dimensions])

    file = 'insurance_experiments.csv'
    with open(file, "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerows(results)
    try:
        upload_file(file)
    except:
        print("File Not Uploaded")

def load_data():
    df = pd.read_csv('insurance.csv')
    return df

if __name__ == '__main__':
    run_i_experiments()

