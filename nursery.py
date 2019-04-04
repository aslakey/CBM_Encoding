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
from dirichlet_encoder import DirichletEncoder
from utils import *
import csv


def run_n_experiments():
    print("Loading Data")
    df = load_data()
    #Creating the dependent variable class
    factor = pd.factorize(df['class'])
    df['class'] = factor[0]
    definitions = factor[1]
    print(df['class'].head())
    print(definitions)

    #columns:
    continuous = []
    categorical = ['parents',
        'has_nurs',
        'form',
        'children',
        'housing',
        'finance',
        'social',
        'health']
    print("continuous columns: ",continuous)
    print("categorical columns: ",categorical)

    X = df[continuous+categorical]
    y = df[['class']]

    models = [
        LogisticRegression(solver='lbfgs',multi_class='multinomial'),
        RandomForestClassifier(n_estimators=100),
        GradientBoostingClassifier(),
        MLPClassifier()]

    results = [['model','Encoder','Accuracy','STD','Training Time','Sparsity', 'Dimensions']]

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

        print("Dirichlet Encoder (mean) Results:")
        acc, f1, time, sparsity, dimensions = cv_binary_classification(model, X, y, continuous, categorical, encoder=DirichletEncoder())
        results.append([type(model), 'DirichletEncoder (m)', acc, f1, time, sparsity, dimensions])


        print("Dirichlet Encoder (mean and variance Results:")
        acc, f1, time, sparsity, dimensions = cv_binary_classification(model, X, y, continuous, categorical, encoder=DirichletEncoder(), moments='mv')
        results.append([type(model), 'DirichletEncoder (mv)', acc, f1, time, sparsity, dimensions])


    file = 'nursery_experiments.csv'
    with open(file, "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerows(results)
    try:
        upload_file(file)
    except:
        print("File Not Uploaded")

def load_data(local=False):
    if not local:
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/nursery/nursery.data"
        r=requests.get(url).content
        df=pd.read_csv(io.StringIO(r.decode('utf-8')),header=None)
        df.to_csv('nursery_raw.csv', index=False)
    else:
        df = pd.read_csv('nursery_raw.csv')

    #rename for readability
    '''
   parents        usual, pretentious, great_pret
   has_nurs       proper, less_proper, improper, critical, very_crit
   form           complete, completed, incomplete, foster
   children       1, 2, 3, more
   housing        convenient, less_conv, critical
   finance        convenient, inconv
   social         non-prob, slightly_prob, problematic
   health         recommended, priority, not_recom
    '''
    names=['parents',
        'has_nurs',
        'form',
        'children',
        'housing',
        'finance',
        'social',
        'health',
        'class']
    new_name = dict(enumerate(names))

    df = df.rename(new_name,axis='columns')

    return df

if __name__ == '__main__':
    run_n_experiments()

