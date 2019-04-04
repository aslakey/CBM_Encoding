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

    # plot increasing dimensionality va computation time
    sample_sizes = [2000,
    4000,
    6000,
    8000,
    10000,
    12000,
    14000,
    16000,
    18000,
    20000,
    22000,
    24000,
    26000,
    28000,
    30000,
    32000,
    34000,
    36000,
    38000,
    40000,
    42000,
    44000,
    46000,
    48000,
    50000]
    sample_sizes = [42000,
    44000,
    46000,
    48000,
    50000]

    results = [['model','Encoder','Accuracy','STD','Training Time','Sparsity','Dimensions','sample_size']]
    for sample_size in sample_sizes:

        print("")
        print("----------------------")
        print("Sample Size: ",sample_size)
        print("----------------------")

        if not sample_size < len(df):
            sample_size = len(df)
        sample = df.sample(sample_size)

        X = sample[continuous+categorical]
        y = sample[['converted']]

        successes = y.sum()[0]
        alpha_prior = float(successes / len(y))

        model = xgb.XGBClassifier(n_jobs=4) #[GradientBoostingClassifier(max_depth=8, n_estimators=64)]

        #BetaEncoder (mean)
        print("Beta Encoder (mean) Results:")
        acc, std, time, sparsity, dimensions = cv_lead_scoring_classification(model, X, y, continuous, categorical, encoder=BetaEncoder(alpha=alpha_prior, beta=1-alpha_prior))
        results.append([type(model), 'BetaEncoder (m)', acc, std, time, sparsity, dimensions,sample_size])

        
        #OneHotEncoder
        print("OneHotEncoder Results:")
        acc, std, time, sparsity, dimensions = cv_lead_scoring_classification(model, X, y, continuous, categorical, encoder=OneHotEncoder(handle_unknown='ignore', sparse=False))
        results.append([type(model), 'OneHotEncoder', acc, std, time, sparsity, dimensions,sample_size])

    file = 'lead_scoring_experiments_comp_time_2.csv'
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

