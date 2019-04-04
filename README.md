# CBM_Encoding

Code for conjugate Bayesian model Encoding.

## Overview

This repository contains code for a potential submission to the 2019 ECML conference.  Python module contain categorical variable encoders using various conjugate prior models.  The encoders are used as base learners in stacked models.  Experiments on several data sets can be ran with the two simple scripts

## Experiments

The paper will consist of two experiments.  The first simply compares BCM encodings with standard encodings. Namely, Ordinal, One-Hot, Binary, Target, and Hashing.  These encoders are implemented using scikit-learn contributed Category Encoders module.  The first experiment entails 3 problem types with two data sets each.  The problem types are binary classification, multi-class classification, and regression which utilize beta encoding, dirichlet encoding, and regression encoding respectively.  Results from each encoded dataset are presented via 4 learning algorithms.  All datasets are publicly available and listed below in the Data section

## Code

Description of the code:

run_experiment_1.py
run_experiment_2.py

## Data
\subsubsection{Adult\protect\footnote{\url{https://archive.ics.uci.edu/ml/machine-learning-databases/adult}.} (Binary Classification)}The Adult dataset was extracted from the 1994 Census Bureau with the task of predicting whether or not someone earns more than \$50,000.  The features are a mix of numeric and categorical.

\subsubsection{Road Safety\protect\footnote{\url{https://data.gov.uk/dataset/cb7ae6f0-4be6-4935-9277-47e5ce24a11f/road-safety-data}.} (Binary Classification)}The Road Safety dataset was utilized by Cerda et. al. in 'Similarity encoding for learning with dirty categorical variables'~\cite{ref_3}.  As in their paper, we randomly sampled 10,000 rows to be used in a binary classification task (target=’Sex of Driver’) with selected categorical features ‘Make’ and ‘Model’.

\subsubsection{Car Evaluation\protect\footnote{\url{https://archive.ics.uci.edu/ml/machine-learning-databases/car}.} (Multiclass Classification)}A decision framework developed by Bohanec and Rajkovic in 1990 to introduce the $\it{Decision}$ ${EXpert}$ (DEX) software package ~\cite{ref_21}. The model evaluates cars according to a mix of categorical and numerical features.

\subsubsection{Nursery\protect\footnote{\url{https://archive.ics.uci.edu/ml/datasets/nursery}.} (Multiclass Classification)}The Nursery dataset is a hierarchical decision model developed in the 1980's to rank nursery school applications. Features are a mix of nominal and ordinal.

\subsubsection{Insurance\protect\footnote{\url{https://github.com/stedy/Machine-Learning-with-R-datasets/blob/master/insurance.csv}.} (Regression)}The insurance dataset originates from Machine Learning with R by Brett Lantz.  The regression problem is to predict medical charges from a set of features.  The dataset is simulated from US Census demographic statistics.

\subsubsection{Bike Sharing\protect\footnote{\url{https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset}.} (Regression)}The Bike Sharing dataset contains hourly demand data from Washington D.C.'s Capital bikeshare program. The task is to predict demand from a rich feature set including weather and holidays.

\subsubsection{PetFinder\protect\footnote{\url{https://www.kaggle.com/c/petfinder-adoption-prediction}} (Multiclass Classification)}This Kaggle competition, hosted by PetFinder, asks Data Scientists to predict the speed at which animals are adopted from a mix of descriptive and online meta features. For this paper, we only use the descriptive, tabular data found in the competition's $\it{train.csv}$ file.

\subsubsection{Lead Scoring\protect\footnote{(pending legal approval) \url{https://github.com/aslakey/CBM_Encoding)}} (Binary Classification)}
The lead scoring data set is a randomly selected sample with a smaller subset of features from WeWork’s actual lead scoring data set.  The task is to predict whether or not a lead books a tour at one of our locations. We have transformed the data to preserve anonymity and mask the actual values of the features themselves.