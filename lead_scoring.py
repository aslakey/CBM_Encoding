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
import xgboost as xgb


def run_ls_experiments():

    print("Loading Data")
    df = load_data()

    continuous = ['company_size', 'interested_desks']
    categorical = ['industry','location', 'lead_source']
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
    256000]

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
    file = 'lead_scoring_dim_experiments.csv'
    with open(file, "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerows(results)
    try:
        upload_file(file)
    except:
        print("File Not Uploaded")

    print('Sample Size: ',len(df))

    X = df[continuous+categorical]
    y = df[['converted']]

    successes = y.sum()[0]
    alpha_prior = float(successes / len(y))

    model = xgb.XGBClassifier(n_jobs=4) #[GradientBoostingClassifier(max_depth=8, n_estimators=64)]

    results = [['model','Encoder','Accuracy','STD','Training Time','Sparsity','Dimensions']]

    print("")
    print("----------------------")
    print("Testing Algorithm: ")
    print(type(model))
    print("----------------------")

    #BetaEncoder (mean)
    print("Beta Encoder (mean) Results:")
    acc, std, time, sparsity, dimensions = cv_lead_scoring_classification(model, X, y, continuous, categorical, encoder=BetaEncoder(alpha=alpha_prior, beta=1-alpha_prior))
    results.append([type(model), 'BetaEncoder (m)', acc, std, time, sparsity, dimensions])

    #BetaEncoder (mean, variance)
    print("Beta Encoder (mean and variance Results:")
    acc, std, time, sparsity, dimensions = cv_lead_scoring_classification(model, X, y, continuous, categorical, encoder=BetaEncoder(alpha=alpha_prior, beta=1-alpha_prior), moments='mv')
    results.append([type(model), 'BetaEncoder (mv)', acc, std, time, sparsity, dimensions])

    file = 'lead_scoring_experiments_official_beta.csv'
    with open(file, "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerows(results)
    try:
        upload_file(file)
    except:
        print("File Not Uploaded")


    #HashingEncoder
    print("HashingEncoder Results:")
    acc, std, time, sparsity, dimensions = cv_lead_scoring_classification(model, X, y, continuous, categorical, encoder=ce.HashingEncoder(return_df=False, n_components=1000))
    results.append([type(model), 'HashingEncoder', acc, std, time, sparsity, dimensions])

    file = 'lead_scoring_experiments_official_hashing.csv'
    with open(file, "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerows(results)
    try:
        upload_file(file)
    except:
        print("File Not Uploaded")
    
    #now truncate categories for one hot
    X = one_hot_truncator(X, categorical, threshold=150, fill={'object':'_other_','number':0})
    
    #OneHotEncoder
    print("OneHotEncoder Results:")
    acc, std, time, sparsity, dimensions = cv_lead_scoring_classification(model, X, y, continuous, categorical, encoder=OneHotEncoder(handle_unknown='ignore', sparse=False))
    results.append([type(model), 'OneHotEncoder', acc, std, time, sparsity, dimensions])

    file = 'lead_scoring_experiments_official.csv'
    with open(file, "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerows(results)
    try:
        upload_file(file)
    except:
        print("File Not Uploaded")

def load_data():
    df = pd.read_csv('lead_scoring_1mil.csv')
    df = df.fillna('null')
    industries = df.industry.str.split(',', n=-1, expand=True)
    df['industry'] = industries[0]
    #training_df['sector'] = industries[1]

    return df

if __name__ == '__main__':
    run_ls_experiments()

