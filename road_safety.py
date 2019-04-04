import pandas as pd
import numpy as np
import io
import requests
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn import metrics
from beta_encoder import BetaEncoder
import category_encoders as ce
from utils import *
import csv


def run_rs_experiments():
    print("Loading Data")
    df = load_data()
    #columns:
    continuous = []
    categorical = ['make','model']

    X = df[continuous+categorical]
    y = df[['Sex_of_Driver']]

    successes = y.sum()[0]
    alpha_prior = float(successes / len(y))

    models = [LogisticRegression(solver='lbfgs'),
        RandomForestClassifier(n_estimators=100),
        GradientBoostingClassifier(),
        MLPClassifier()]

    results = [['model','Encoder','Accuracy','STD','Training Time','Sparsity','Dimensions']]

    for model in models:
        print("")
        print("----------------------")
        print("Testing Algorithm: ")
        print(type(model))
        print("----------------------")

        #TargetEncoder
        print("TargetEncoder Results:")
        acc, std, time, sparsity, dimensions = cv_binary_classification(model, X, y, continuous, categorical, encoder=ce.TargetEncoder(return_df=False))
        results.append([type(model), 'TargetEncoder', acc, std, time, sparsity, dimensions])

        #OrdinalEncoder
        print("OrdinalEncoder Results:")
        acc, std, time, sparsity, dimensions = cv_binary_classification(model, X, y, continuous, categorical, encoder=ce.OrdinalEncoder(return_df=False))
        results.append([type(model), 'OrdinalEncoder', acc, std, time, sparsity, dimensions])

        #BinaryEncoder
        print("BinaryEncoder Results:")
        acc, std, time, sparsity, dimensions = cv_binary_classification(model, X, y, continuous, categorical, encoder=ce.BinaryEncoder(return_df=False))
        results.append([type(model), 'BinaryEncoder', acc, std, time, sparsity, dimensions])

        #HashingEncoder
        print("HashingEncoder Results:")
        acc, std, time, sparsity, dimensions = cv_binary_classification(model, X, y, continuous, categorical, encoder=ce.HashingEncoder(return_df=False))
        results.append([type(model), 'HashingEncoder', acc, std, time, sparsity, dimensions])

        #OneHotEncoder
        print("OneHotEncoder Results:")
        acc, std, time, sparsity, dimensions = cv_binary_classification(model, X, y, continuous, categorical, encoder=OneHotEncoder(handle_unknown='ignore', sparse=False))
        results.append([type(model), 'OneHotEncoder', acc, std, time, sparsity, dimensions])

        #BetaEncoder (mean)
        print("Beta Encoder (mean) Results:")
        acc, std, time, sparsity, dimensions = cv_binary_classification(model, X, y, continuous, categorical, encoder=BetaEncoder(alpha=alpha_prior, beta=1-alpha_prior))
        results.append([type(model), 'BetaEncoder (m)', acc, std, time, sparsity, dimensions])

        #BetaEncoder (mean, variance)
        print("Beta Encoder (mean and variance Results:")
        acc, std, time, sparsity, dimensions = cv_binary_classification(model, X, y, continuous, categorical, encoder=BetaEncoder(alpha=alpha_prior, beta=1-alpha_prior), moments='mv')
        results.append([type(model), 'BetaEncoder (mv)', acc, std, time, sparsity, dimensions])


    file = 'road_safety_experiments.csv'
    with open(file, "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerows(results)

    try:
        upload_file(file)
    except:
        print("File Not Uploaded")

def load_data():
    df = pd.read_csv('road_safety_raw.csv')
    df = df.sample(10000)
    df.to_csv('road_safety_raw.csv', index=False)

    return df

if __name__ == '__main__':
    run_rs_experiments()

