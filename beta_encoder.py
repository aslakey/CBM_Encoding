import pandas as pd
import numpy as np

class BetaEncoder:
    '''BetaEncoder

    BetaEncoder is used to encode categorical features with a beta-binomial conjugate pair
    model (i.e. a beta posterior predictive distribution for binary target). 
    For each categorical feature, this object stores a beta (y==0) and alpha (y==1)
    column with a row for each existing level of the categorical feature.  

    The input to fit() should be an array-like of 1,0 for y
    and array-like of strings for X.
    The output of transform() will be <column>__[M] where [M] is a particular moment 
    of the beta distribution [‘mvsk’], m and v are default.
    By default, a prior of alpha=.5, beta=.5 (uninformative) is used.

    Note: transform() takes the optional argument `training` (bool) for which
    y must be supplied. This results in total_count being decremented and 
    alpha or beta being decremented depending on y value.

    Parameters
    ----------
    beta_prior (float): prior for beta. default = .5
    alpha_prior (float):  prior for alpha. default = .5
    random_state (integer): random state for bootstrap samples. default = 1
    n_samples (integer): number of bootstrap samples. default = 100

    Attributes
    ----------
    _beta_prior (float) - prior for beta. default = .5
    _alpha_prior (float) - prior for alpha. default = .5
    _random_state (integer): random state for bootstrap samples. default = 1
    _n_samples (integer): number of bootstrap samples. default = 100
    _beta_distributions (dict) - houses the categorical beta distributions
        in pandas dataframes with cols `alpha` and `beta`


    Methods
    ----------
    fit()
    transform()


    Examples
    --------
    >>>import pandas as pd
    >>>from sklearn.datasets import load_boston
    >>>from sklearn.model_selection import train_test_split
    >>>from beta_encoder import BetaEncoder
    >>>bunch = load_boston()
    >>>y = bunch.target
    >>>X = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    >>>categorical_cols=['CHAS', 'RAD']
    >>>X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=44)
    >>>encoder = BetaEncoder()
    >>>encoder.fit(X_train, y_train, columns=categorical_cols)
    >>>#print out the beta parameters for each level
    >>>encoder._beta_distributions
    >>>#transform the training dataset (leave one out)
    >>>encoder.transform(X_train, y=y_train, training=True, columns=categorical_cols)
    >>>#transform the test columns (just a pure link and fill na with prior)
    >>>encoder.transform(X_test, columns=categorical_cols)

    '''
    def __init__(self, alpha=0.5, beta=0.5, n_samples=10, sample_size=.75, random_state=1):
        '''init for BetaEncoder
        Args:
            alpha - prior for number of successes
            beta - prior for number of failures

        '''
        # Validate Types
        if type(alpha) != float:
            raise AttributeError("Argument 'alpha' must be of type float")
        if type(beta) is not float:
            raise AttributeError("Argument 'beta' must be of type float")
        if type(sample_size) is not float:
            raise AttributeError("Argument 'sample_size' must be of type float")
        if type(n_samples) is not int:
            raise AttributeError("Argument 'n_samples' must be of type int")
        if type(random_state) is not int:
            raise AttributeError("Argument 'random_state' must be of type int")

        #Assign
        self._alpha_prior = alpha
        self._beta_prior = beta
        self._beta_distributions = dict()
        self._random_state = random_state
        self._n_samples = n_samples
        self._sample_size = sample_size
        np.random.seed(random_state)

    def fit(self, X, y, columns=None):
        '''fit
        Method to fit self.beta_distributions
        from X and y

        Args:
            X (array-like) - categorical columns
            y (array-like) - target column (1,0)
            columns (list of str) - list of column names to fit
                otherwise, attempt to fit just string columns
        Returns:
            beta_distributions (dict) - a dict of pandas DataFrame for each
                categorical column with beta and alpha for each level
        '''
        if len(X) != len(y):
            print("received: ",len(X), len(y))
            raise AssertionError("Length of X and y must be equal.")

        X_temp = X.copy(deep=True)
        categorical_cols = columns
        if not categorical_cols:
            categorical_cols = self.get_string_cols(X_temp)

        #add target
        target_col = '_target'
        X_temp[target_col] = y


        for categorical_col in categorical_cols:

            # All Levels
            #   Bootstrap samples may not contain all levels, so fill NA with priors
            ALL_LEVELS = X_temp[[categorical_col, target_col]].groupby(categorical_col).count().reset_index()

            for i in range(self._n_samples):

                X_sample = X_temp[[categorical_col, target_col]].sample(n=int(len(X_temp)*self._sample_size), replace=True, random_state=self._random_state + i)
                
                #full count (alpha + beta)
                full_count = X_sample[[categorical_col, target_col]].groupby(categorical_col).count().reset_index()
                full_count = full_count.rename(index=str, columns={target_col: categorical_col+"_full_count"})

                #alpha
                positive_count = X_sample[[categorical_col, target_col]].groupby(categorical_col).sum().reset_index()
                positive_count = positive_count.rename(index=str, columns={target_col: categorical_col+"_positive_count"})
                
                #merge them
                temp = pd.merge(full_count, positive_count, on=[categorical_col])
                temp['_alpha'] = self._alpha_prior + temp[categorical_col+"_positive_count"]
                temp['_beta'] = self._beta_prior + temp[categorical_col+"_full_count"] - temp[categorical_col+"_positive_count"]

                #fill NAs with prior
                temp = pd.merge(ALL_LEVELS, temp, on=categorical_col, how='left')
                temp['_alpha'] = temp['_alpha'].fillna(self._alpha_prior)
                temp['_beta'] = temp['_beta'].fillna(self._beta_prior)

                if categorical_col not in self._beta_distributions.keys():
                    self._beta_distributions[categorical_col] = temp[[categorical_col,'_alpha','_beta']]
                else:
                    self._beta_distributions[categorical_col][['_alpha','_beta']] += temp[['_alpha','_beta']]

            # report mean alpha and beta:
            self._beta_distributions[categorical_col]['_alpha'] = self._beta_distributions[categorical_col]['_alpha']/self._n_samples
            self._beta_distributions[categorical_col]['_beta'] = self._beta_distributions[categorical_col]['_beta']/self._n_samples
        return

    def transform(self, X, moments='m', columns=None):
        '''transform
        Args:
            X (array-like) - categorical columns matching
                the columns in beta_distributions
            columns (list of str) - list of column names to transform
                otherwise, attempt to transform just string columns
            moments (str) - composed of letters [‘mvsk’] 
                specifying which moments to compute where ‘m’ = mean, 
                ‘v’ = variance, ‘s’ = (Fisher’s) skew and ‘k’ = (Fisher’s) 
                kurtosis. (default=’m’)
        '''
        X_temp = X.copy(deep=True)
        categorical_cols = columns
        if not categorical_cols:
            categorical_cols = self.get_string_cols(X_temp)


        for categorical_col in categorical_cols:
            if categorical_col not in self._beta_distributions.keys():
                raise AssertionError("Column "+categorical_col+" not fit by BetaEncoder")

            #add `_alpha` and `_beta` columns vi lookups, impute with prior
            X_temp = X_temp.merge(self._beta_distributions[categorical_col], on=[categorical_col], how='left')

            X_temp['_alpha'] = X_temp['_alpha'].fillna(self._alpha_prior)
            X_temp['_beta'] = X_temp['_beta'].fillna(self._beta_prior)         
            
            #   encode with moments
            if 'm' in moments:
                X_temp[categorical_col+'__M'] = X_temp["_alpha"]/(X_temp["_alpha"]+X_temp["_beta"])
            if 'v' in moments:
                X_temp[categorical_col+'__V'] = (X_temp["_alpha"]*X_temp["_beta"]) / \
                   (((X_temp["_alpha"]+X_temp["_beta"])**2)*(X_temp["_alpha"]+X_temp["_beta"]+1))
            #and drop columns
            X_temp = X_temp.drop([categorical_col], axis=1)
            X_temp = X_temp.drop(["_alpha"], axis=1)
            X_temp = X_temp.drop(["_beta"], axis=1)

        return X_temp

    def get_string_cols(self, df):
        idx = (df.applymap(type) == str).all(0)
        return df.columns[idx]