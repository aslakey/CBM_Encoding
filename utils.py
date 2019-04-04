import numpy as np
import pandas as pd
from beta_encoder import BetaEncoder
from dirichlet_encoder import DirichletEncoder
from gaussian_inverse_gamma_encoder import GIGEncoder
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn import preprocessing
from timeit import default_timer as timer
import category_encoders as ce
import copy
import boto3
import os
import sys
import io

def sparsity_ratio(X):
    '''sparsity_ration
    # of zero entries / total Matrix size
    Args:
        X (numpy matrix) - a 2-D numpy matrix

    Returns:
        sparsity_ratio (float)
    '''
    #print("input sparsity ratio:", sparsity_ratio(X))
    return 1.0 - (np.count_nonzero(X) / (X.shape[0] * X.shape[1]))

def cv_binary_classification(model, X, y, continuous, categorical, encoder=OneHotEncoder(handle_unknown='ignore', sparse=False), moments='m', n_splits = 10):
    '''Cross Validation Code
    Args:
        model - an sklearn model with .predict() and .fit() methods
        X - data frame with just feature cols
        y - data frame with just target col
        continuous - list of continuous columns
        categorical - list of categorical columns
        encoder (encoder object) - SKlearn-style categorical variable encoder
        
    Returns: cross validated average for...
        ACC - mean accuracy = 1/N 
        F1 - mean F1 score
            where 
                F1 = (2* precision * recall) / (precision + recall)
                precision = tp/ (tp + fp)
                recall = tp / (tp + fn)
        training_time - average final model training time
    '''
    ACC = np.zeros(n_splits)
    TRAIN_ACC = np.zeros(n_splits)
    model_training_time = 0
    sparsity = 0
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=1)
    kf.get_n_splits(X)
    i = -1
    for train_index, test_index in kf.split(X):
        i+=1
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        #fit encoders and scalers on training data
        enc = copy.deepcopy(encoder)
        scaler = StandardScaler()

        # fit timer start
        start = timer()


        if type(enc) is OneHotEncoder:
            enc.fit(X_train[categorical])
        elif (type(enc) is BetaEncoder) or (type(enc) is DirichletEncoder):
            enc.fit(X_train[categorical], y_train, columns=categorical)
        #sklearn ce objects
        else:
            enc.fit(X_train[categorical], y_train, cols=categorical)
        
        # scale
        if len(continuous) > 0:
            scaler.fit(X_train[continuous].astype(float))
        
        # transform the categorical columns
        if type(enc) is OneHotEncoder:
            X_train_categorical_cols = enc.transform(X_train[categorical],)
            X_test_categorical_cols = enc.transform(X_test[categorical])
        elif type(enc) is BetaEncoder or (type(enc) is DirichletEncoder):
            X_train_categorical_cols = enc.transform(X_train[categorical], moments=moments, columns=categorical)
            X_test_categorical_cols = enc.transform(X_test[categorical], moments=moments, columns=categorical)
        #sklearn ce objects
        else:
            if type(enc) is ce.TargetEncoder:
                X_train_categorical_cols = enc.fit_transform(X_train[categorical].reset_index(drop=True),y_train.reset_index(drop=True))      
                X_test_categorical_cols = enc.transform(X_test[categorical].reset_index(drop=True), y=None)
            else:
                X_train_categorical_cols = enc.transform(X_train[categorical])
                X_test_categorical_cols = enc.transform(X_test[categorical])
        
        # scale continuous
        if len(continuous) > 0:
            X_train_cont_cols = scaler.transform(X_train[continuous].astype(float))
            X_test_cont_cols = scaler.transform(X_test[continuous].astype(float))
        
        # Concatenate (Column-Bind) Processed Columns Back Together
        if len(continuous) > 0:
            X_train = np.concatenate([X_train_categorical_cols, X_train_cont_cols], axis=1)
            X_test = np.concatenate([X_test_categorical_cols, X_test_cont_cols], axis=1)
        else:
            X_train = X_train_categorical_cols
            X_test = X_test_categorical_cols

        # calculate sparsity and dims
        sparsity = sparsity_ratio(X_train)
        dimensions = X_train.shape[1]
        
        model.fit(X_train, y_train.values.ravel())
        end = timer()
        #time in seconds
        model_training_time += end - start
        
        #training data
        y_pred = model.predict(X_train)
        TRAIN_ACC[i] = accuracy_score(y_train, y_pred)

        # Predict on new data
        y_pred = model.predict(X_test)
        ACC[i] = accuracy_score(y_test, y_pred)

    print("")
    print("----------------")
    print("CV Results")
    print("Encoder: ", type(encoder))
    print("Model: ", type(model))
    print("----------------")
    print("Training Accuracy: ", np.mean(TRAIN_ACC))
    print("Accuracy: ", np.mean(ACC))
    print("STD: ", np.std(ACC))
    print("Training Time: ",model_training_time/n_splits)
    print("Sparsity: ",sparsity)
    print("Dimensions: ",dimensions)
    print("")


    return (np.mean(ACC), np.std(ACC), model_training_time/n_splits, sparsity, dimensions)

def cv_regression(model, X, y, continuous, categorical, encoder=OneHotEncoder(handle_unknown='ignore', sparse=False), moments='m', n_splits = 10):
    '''Cross Validation Code
    Args:
        model - an sklearn model with .predict() and .fit() methods
        X - data frame with just feature cols
        y - data frame with just target col
        continuous - list of continuous columns
        categorical - list of categorical columns
        encoder (encoder object) - SKlearn-style categorical variable encoder
        
    Returns: cross validated average for...
        ACC - mean accuracy = 1/N 
        F1 - mean F1 score
            where 
                F1 = (2* precision * recall) / (precision + recall)
                precision = tp/ (tp + fp)
                recall = tp / (tp + fn)
        training_time - average final model training time
    '''
    r2 = np.zeros(n_splits)
    model_training_time = 0
    sparsity = 0
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=1)
    kf.get_n_splits(X)
    i = -1
    for train_index, test_index in kf.split(X):
        i+=1
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        #fit encoders and scalers on training data
        enc = copy.deepcopy(encoder)
        scaler = StandardScaler()
        if type(enc) is OneHotEncoder:
            enc.fit(X_train[categorical])
        elif (type(enc) is GIGEncoder):
            enc.fit(X_train[categorical], y_train, columns=categorical)
        #sklearn ce objects
        else:
            enc.fit(X_train[categorical], y_train, cols=categorical)
        
        # scale
        if len(continuous) > 0:
            scaler.fit(X_train[continuous].astype(float))
        
        # transform the categorical columns
        if type(enc) is OneHotEncoder:
            X_train_categorical_cols = enc.transform(X_train[categorical],)
            X_test_categorical_cols = enc.transform(X_test[categorical])
        elif type(enc) is GIGEncoder:
            X_train_categorical_cols = enc.transform(X_train[categorical], moments=moments, columns=categorical)
            X_test_categorical_cols = enc.transform(X_test[categorical], moments=moments, columns=categorical)
        #sklearn ce objects
        else:
            if type(enc) is ce.TargetEncoder:
                X_train_categorical_cols = enc.fit_transform(X_train[categorical].reset_index(drop=True),y_train.reset_index(drop=True))      
                X_test_categorical_cols = enc.transform(X_test[categorical].reset_index(drop=True), y=None)
            else:
                X_train_categorical_cols = enc.transform(X_train[categorical])
                X_test_categorical_cols = enc.transform(X_test[categorical])
        
        # scale continuous
        if len(continuous) > 0:
            X_train_cont_cols = scaler.transform(X_train[continuous].astype(float))
            X_test_cont_cols = scaler.transform(X_test[continuous].astype(float))
        
        # Concatenate (Column-Bind) Processed Columns Back Together
        if len(continuous) > 0:
            X_train = np.concatenate([X_train_categorical_cols, X_train_cont_cols], axis=1)
            X_test = np.concatenate([X_test_categorical_cols, X_test_cont_cols], axis=1)
        else:
            X_train = X_train_categorical_cols
            X_test = X_test_categorical_cols

        # calculate sparsity and dims
        sparsity = sparsity_ratio(X_train)
        dimensions = X_train.shape[1]
        
        # fit
        start = timer()
        model.fit(X_train, y_train.values.ravel())
        end = timer()
        #time in seconds
        model_training_time += end - start
        

        # Predict on new data
        y_pred = model.predict(X_test)
        r2[i] = metrics.r2_score(y_test, y_pred)

    print("")
    print("----------------")
    print("CV Results")
    print("Encoder: ", type(encoder))
    print("Model: ", type(model))
    print("----------------")
    print("r2: ", np.mean(r2))
    print("STD: ", np.std(r2))
    print("Training Time: ",model_training_time/n_splits)
    print("Sparsity: ",sparsity)
    print("Dimensions: ",dimensions)
    print("")


    return (np.mean(r2), np.std(r2), model_training_time/n_splits, sparsity, dimensions)

def upload_file(filename, bucket='wework-growth-analytics', directory='KDD/'):
    '''upload_file
    uploads a local file to s3 bucket.
    Attempts assume role (if in another AWS account), on error it
    attempts default credentials (on our AWS account)

    Args:
        filename (str) - name of file
    Returns:
        None
    Raises:
        None
    '''
    try:
        print("Attempting AWS role")
        s3 = sts_assume_role_s3()
    except:
        print("Using default aws creds")
        s3 = boto3.resource('s3')
    s3.Bucket(bucket).upload_file(filename, directory+filename, ExtraArgs={'ACL':'bucket-owner-full-control'})
    return

def sts_assume_role_s3():
    '''sts_assume_role_s3
    tries to assume a role in the data analytics account
    that allows full s3 access.  It is only accessible from the ECS
    Service in the rex account.

    Args:
        None
    Returns:
        s3_resource - a boto3.resource('s3') using the IAM role credentials
        from rex-growth-analytics-s3-full
    '''

    sts_client = boto3.client('sts')
    assumedRoleObject = sts_client.assume_role(
        RoleArn="arn:aws:iam::247004163247:role/rex-growth-analytics-s3-full",
        RoleSessionName="AssumeRoleFromRex"
    )

    # From the response that contains the assumed role, get the temporary 
    # credentials that can be used to make subsequent API calls
    credentials = assumedRoleObject['Credentials']

    # Use the temporary credentials that AssumeRole returns to make a 
    # connection to Amazon S3  
    s3_resource = boto3.resource(
        's3',
        aws_access_key_id = credentials['AccessKeyId'],
        aws_secret_access_key = credentials['SecretAccessKey'],
        aws_session_token = credentials['SessionToken'],
    )

    return s3_resource

def cv_lead_scoring_classification(model, X, y, continuous, categorical, encoder=OneHotEncoder(handle_unknown='ignore', sparse=False), moments='m', n_splits = 5):
    '''Cross Validation Code for Lead Scoring

    Accuracy is AUC.  5 Fold Cross Validation
    Args:
        model - an sklearn model with .predict() and .fit() methods
        X - data frame with just feature cols
        y - data frame with just target col
        continuous - list of continuous columns
        categorical - list of categorical columns
        encoder (encoder object) - SKlearn-style categorical variable encoder
        
    Returns: cross validated average for...
        ACC - mean accuracy = 1/N 
        F1 - mean F1 score
            where 
                F1 = (2* precision * recall) / (precision + recall)
                precision = tp/ (tp + fp)
                recall = tp / (tp + fn)
        training_time - average final model training time
    '''
    ACC = 0
    model_training_time = 0
    sparsity = 0

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

    #fit encoders and scalers on training data
    enc = copy.deepcopy(encoder)
    scaler = StandardScaler()

    # fit
    start = timer()
    print("encoding")
    if type(enc) is OneHotEncoder:
        enc.fit(X_train[categorical])
    elif (type(enc) is BetaEncoder) or (type(enc) is DirichletEncoder):
        enc.fit(X_train[categorical], y_train, columns=categorical)
    #sklearn ce objects
    else:
        enc.fit(X_train[categorical], y_train, cols=categorical)
    
    # scale
    print("scaling")
    if len(continuous) > 0:
        scaler.fit(X_train[continuous].astype(float))
    
    # transform the categorical columns
    print("transforming")
    if type(enc) is OneHotEncoder:
        X_train_categorical_cols = enc.transform(X_train[categorical],)
        X_test_categorical_cols = enc.transform(X_test[categorical])
    elif type(enc) is BetaEncoder or (type(enc) is DirichletEncoder):
        X_train_categorical_cols = enc.transform(X_train[categorical], moments=moments, columns=categorical)
        X_test_categorical_cols = enc.transform(X_test[categorical], moments=moments, columns=categorical)
    #sklearn ce objects
    else:
        if type(enc) is ce.TargetEncoder:
            X_train_categorical_cols = enc.fit_transform(X_train[categorical].reset_index(drop=True),y_train.reset_index(drop=True))      
            X_test_categorical_cols = enc.transform(X_test[categorical].reset_index(drop=True), y=None)
        else:
            X_train_categorical_cols = enc.transform(X_train[categorical])
            X_test_categorical_cols = enc.transform(X_test[categorical])
    
    # scale continuous
    if len(continuous) > 0:
        X_train_cont_cols = scaler.transform(X_train[continuous].astype(float))
        X_test_cont_cols = scaler.transform(X_test[continuous].astype(float))
    
    # Concatenate (Column-Bind) Processed Columns Back Together
    if len(continuous) > 0:
        X_train = np.concatenate([X_train_categorical_cols, X_train_cont_cols], axis=1)
        X_test = np.concatenate([X_test_categorical_cols, X_test_cont_cols], axis=1)
    else:
        X_train = X_train_categorical_cols
        X_test = X_test_categorical_cols

    # calculate sparsity and dims
    sparsity = sparsity_ratio(X_train)
    dimensions = X_train.shape[1]
    print("dimensions: ",dimensions)
    
    model.fit(X_train, y_train.values.ravel())
    end = timer()
    #time in seconds
    model_training_time += end - start
    

    # Predict probabilities (positive label) on new data
    y_pred = model.predict_proba(X_test)[:,1]
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    ACC = metrics.auc(fpr, tpr)

    print("")
    print("----------------")
    print("CV Results")
    print("Encoder: ", type(encoder))
    print("Model: ", type(model))
    print("----------------")
    print("Accuracy: ", ACC)
    print("STD: ", ACC)
    print("Training Time: ",model_training_time)
    print("Sparsity: ",sparsity)
    print("Dimensions: ",dimensions)
    print("")


    return (ACC, ACC, model_training_time, sparsity, dimensions)

def cv_petfinder_classification(model, X, y, continuous, categorical, encoder=OneHotEncoder(handle_unknown='ignore', sparse=False), moments='m', n_splits = 5):
    '''Cross Validation Code for Pet Finder

    Accuracy is Quadratic Weighted Kappa.  5 Fold Cross Validation
    Args:
        model - an sklearn model with .predict() and .fit() methods
        X - data frame with just feature cols
        y - data frame with just target col
        continuous - list of continuous columns
        categorical - list of categorical columns
        encoder (encoder object) - SKlearn-style categorical variable encoder
        
    Returns: cross validated average for...
        ACC - mean accuracy = 1/N 
        F1 - mean F1 score
            where 
                F1 = (2* precision * recall) / (precision + recall)
                precision = tp/ (tp + fp)
                recall = tp / (tp + fn)
        training_time - average final model training time
    '''
    ACC = 0
    model_training_time = 0
    sparsity = 0

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

    #fit encoders and scalers on training data
    enc = copy.deepcopy(encoder)
    scaler = StandardScaler()

    # fit
    start = timer()

    if type(enc) is OneHotEncoder:
        enc.fit(X_train[categorical])
    elif (type(enc) is BetaEncoder) or (type(enc) is DirichletEncoder) or (type(enc) is GIGEncoder):
        enc.fit(X_train[categorical], y_train, columns=categorical)
    #sklearn ce objects
    else:
        enc.fit(X_train[categorical], y_train, cols=categorical)
    
    # scale
    if len(continuous) > 0:
        scaler.fit(X_train[continuous].astype(float))
    
    # transform the categorical columns
    if type(enc) is OneHotEncoder:
        X_train_categorical_cols = enc.transform(X_train[categorical],)
        X_test_categorical_cols = enc.transform(X_test[categorical])
    elif type(enc) is BetaEncoder or (type(enc) is DirichletEncoder) or (type(enc) is GIGEncoder):
        X_train_categorical_cols = enc.transform(X_train[categorical], moments=moments, columns=categorical)
        X_test_categorical_cols = enc.transform(X_test[categorical], moments=moments, columns=categorical)
    #sklearn ce objects
    else:
        if type(enc) is ce.TargetEncoder:
            X_train_categorical_cols = enc.fit_transform(X_train[categorical].reset_index(drop=True),y_train.reset_index(drop=True))      
            X_test_categorical_cols = enc.transform(X_test[categorical].reset_index(drop=True), y=None)
        else:
            X_train_categorical_cols = enc.transform(X_train[categorical])
            X_test_categorical_cols = enc.transform(X_test[categorical])
    
    # scale continuous
    if len(continuous) > 0:
        X_train_cont_cols = scaler.transform(X_train[continuous].astype(float))
        X_test_cont_cols = scaler.transform(X_test[continuous].astype(float))
    
    # Concatenate (Column-Bind) Processed Columns Back Together
    if len(continuous) > 0:
        X_train = np.concatenate([X_train_categorical_cols, X_train_cont_cols], axis=1)
        X_test = np.concatenate([X_test_categorical_cols, X_test_cont_cols], axis=1)
    else:
        X_train = X_train_categorical_cols
        X_test = X_test_categorical_cols

    # calculate sparsity and dims
    sparsity = sparsity_ratio(X_train)
    dimensions = X_train.shape[1]
    print("dimensions: ",dimensions)
    

    model.fit(X_train, y_train.values.ravel())
    end = timer()
    #time in seconds
    model_training_time += end - start
    

    # Predict labels on new data
    y_pred = model.predict(X_test)
    ACC = metrics.cohen_kappa_score(y_test, y_pred, weights='quadratic')

    print("")
    print("----------------")
    print("CV Results")
    print("Encoder: ", type(encoder))
    print("Model: ", type(model))
    print("----------------")
    print("Accuracy: ", ACC)
    print("STD: ", ACC)
    print("Training Time: ",model_training_time)
    print("Sparsity: ",sparsity)
    print("Dimensions: ",dimensions)
    print("")


    return (ACC, ACC, model_training_time, sparsity, dimensions)

def one_hot_truncator(df, columns, threshold=100, fill={'object':'_other_','number':0}):
    
    for col in columns:
        
        filler = fill['number']
        if df[col].dtype == 'object':
            filler = fill['object']
        df.loc[df.groupby(col)[col].transform('count').lt(threshold), col] = filler 


    return df


