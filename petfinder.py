import pandas as pd
import numpy as np
import io
import requests
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn import metrics
from dirichlet_encoder import DirichletEncoder
from gaussian_inverse_gamma_encoder import GIGEncoder
import category_encoders as ce
from utils import *
import csv


def run_pf_experiments():

    print("Loading Data")
    df = load_data()

    continuous = ['Age','Fee', 'Quantity','VideoAmt', 'PhotoAmt','MaturitySize', 'FurLength', 'Health']

    categorical = ['Type', 'Breed1', 'Breed2','Gender','Color1','Color2','Color3',
    'Vaccinated', 'Dewormed', 'Sterilized',
    'State', 'RescuerID']
    
    #columns:
    print("continuous columns: ",continuous)
    print("categorical columns: ",categorical)

    #first plot increasing dimensionality
    sample_sizes = [2000,
    4000,
    8000,
    16000,
    64000,
    128000,
    256000,
    512000,
    1024000]

    results = [['Sample Size','Dimensions']]
    for sample_size in sample_sizes:
        print("Sample Size: ",sample_size)
        if not sample_size < len(df):
            sample_size = len(df)
        sample = df.sample(sample_size)
        dims = 0
        for col in categorical:
            dims += len(sample[col].unique())

        results.append([sample_size, dims])
    
    #output file
    file = 'pet_finder_dim_experiments.csv'
    with open(file, "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerows(results)
    try:
        upload_file(file)
    except:
        print("File Not Uploaded")

    X = df[continuous+categorical]
    y = df[['AdoptionSpeed']]

    successes = y.sum()[0]
    alpha_prior = float(successes / len(y))

    model = xgb.XGBClassifier(n_jobs=4)

    results = [['model','Encoder','Accuracy','STD','Training Time','Sparsity','Dimensions']]

    print("")
    print("----------------------")
    print("Testing Algorithm: ")
    print(type(model))
    print("----------------------")

    print("GIG Encoder (mean) Results:")
    r2, std, time, sparsity, dimensions = cv_petfinder_classification(model, X, y, continuous, categorical, encoder=GIGEncoder(u_0=2.5, v=1.0))
    results.append([type(model), 'GIGEncoder (m)', r2, std, time, sparsity, dimensions])
    
    #HashingEncoder
    print("HashingEncoder Results:")
    acc, std, time, sparsity, dimensions = cv_petfinder_classification(model, X, y, continuous, categorical, encoder=ce.HashingEncoder(return_df=False, n_components=1000))
    results.append([type(model), 'HashingEncoder', acc, std, time, sparsity, dimensions])

    print("Dirichlet Encoder (mean) Results:")
    acc, f1, time, sparsity, dimensions = cv_petfinder_classification(model, X, y, continuous, categorical, encoder=DirichletEncoder())
    results.append([type(model), 'DirichletEncoder (m)', acc, f1, time, sparsity, dimensions])

    #now truncate categories for one hot
    X = one_hot_truncator(X, categorical, threshold=10, fill={'object':'_other_','number':0})

    #OneHotEncoder
    print("OneHotEncoder Results:")
    acc, std, time, sparsity, dimensions = cv_petfinder_classification(model, X, y, continuous, categorical, encoder=OneHotEncoder(handle_unknown='ignore', sparse=False))
    results.append([type(model), 'OneHotEncoder', acc, std, time, sparsity, dimensions])


    file = 'pet_finder_experiments.csv'
    with open(file, "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerows(results)
    try:
        upload_file(file)
    except:
        print("File Not Uploaded")

def load_data():
    df = pd.read_csv('petfinder_train.csv')
    df = df.fillna('null')
    return df

if __name__ == '__main__':
    run_pf_experiments()

