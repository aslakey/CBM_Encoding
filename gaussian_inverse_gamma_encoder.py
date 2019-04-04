import pandas as pd
import numpy as np

class GIGEncoder:
    '''NormalInverseGammaEncoder

    GaussianInverseGammaEncoder is used to encode categorical features with a Normal - GIG 
    conjugate pair model (i.e. a Normal Inverse Gamma posterior distribution for a normally
    distributed target).  https://en.wikipedia.org/wiki/Normal-inverse-gamma_distribution

    For each categorical feature, this object stores a updated parameters mu, v, alpha, beta
    columns with a row for each existing level of the categorical feature.

    interpretation:
    mean was estimated from {\displaystyle \nu } \nu 
    observations with sample mean {\displaystyle \mu _{0}} \mu _{0}; 
    variance was estimated from {\displaystyle 2\alpha } 2\alpha 
    observations with sample mean {\displaystyle \mu _{0}} \mu _{0} 
    and sum of squared deviations {\displaystyle 2\beta } 2\beta   

    The input to fit() should be an array-like of real numbers for y
    and array-like of strings for X.
    The output of transform() will be <column>__[M]_[X | V] where [M] is a particular moment 
    of the normal inverse gamma distribution [‘mvsk’] (m and v are default) and X or sigma refers to 
    expected value of X or Variance.
    By default, a prior of u_0=0, v=1, alpha =2, beta=1  (Standard Gaussian) is used.

    Parameters
    ----------
    u_0 (float): prior. default = 0
    v (float): prior. default = 1
    beta (float): prior for beta. default = 1
    alpha (float):  prior for alpha. default = 2
    random_state (integer): random state for bootstrap samples. default = 1
    n_samples (integer): number of bootstrap samples. default = 100

    Attributes
    ----------
    _u_0_prior (float): prior. default = 0
    _v_prior (float): prior. default = 1
    _beta_prior (float) - prior for beta. default = .5
    _alpha_prior (float) - prior for alpha. default = .5
    _random_state (integer): random state for bootstrap samples. default = 1
    _n_samples (integer): number of bootstrap samples. default = 100
    _gig_distributions (dict) - houses the categorical beta distributions
        in pandas dataframes with cols `alpha` and `beta`


    Methods
    ----------
    fit()
    transform()


    Examples
    --------
    >>>import pandas as pd
    >>>import pymc3 as pm
    >>>from normal_inverse_gamma_encoder import GIGEncoder

    >>>data = pd.read_csv(pm.get_data('radon.csv'))
    >>>data = data[['county', 'log_radon', 'floor']]
    >>>enc = GIGEncoder()
    >>>enc.fit(data[['county', 'floor']],data['log_radon'],['county', 'floor'])
    >>>enc._gig_distributions
    >>>enc.transform(data[['county', 'floor']],columns=['county', 'floor']).head()

    '''
    def __init__(self, u_0=0.0, v=1.0, alpha=3.0, beta=1.0, n_samples=10, sample_size=.75, random_state=1):
        '''init for BetaEncoder
        Args:
            alpha - prior for number of successes
            beta - prior for number of failures

        '''
        # Validate Types
        if type(u_0) is not float and type(u_0) is not np.float64:
            raise AttributeError("Argument 'u_0' must be of type float")
        if type(v) is not float and type(v) is not np.float64:
            raise AttributeError("Argument 'v' must be of type float")
        if type(alpha) != float:
            raise AttributeError("Argument 'alpha' must be of type float")
        if type(beta) is not float:
            raise AttributeError("Argument 'beta' must be of type float")
        if type(n_samples) is not int:
            raise AttributeError("Argument 'n_samples' must be of type int")
        if type(random_state) is not int:
            raise AttributeError("Argument 'random_state' must be of type int")

        #Assign
        self._alpha_prior = alpha
        self._beta_prior = beta
        self._u_0_prior = u_0
        self._v_prior = v
        self._gig_distributions = dict()
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
            y (array-like) - target column (float))
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
                
                #full count (n)
                full_count = X_sample[[categorical_col, target_col]].groupby(categorical_col).count().reset_index()
                full_count = full_count.rename(index=str, columns={target_col: categorical_col+"_full_count"})

                #mean
                sample_mean = X_sample[[categorical_col, target_col]].groupby(categorical_col).mean().reset_index()
                sample_mean = sample_mean.rename(index=str, columns={target_col: categorical_col+"_sample_mean"})

                #std
                std = X_sample[[categorical_col, target_col]].groupby(categorical_col).std().reset_index()
                std = std.rename(index=str, columns={target_col: categorical_col+"_std"})
                
                #merge them
                temp = pd.merge(full_count, sample_mean, on=[categorical_col])
                temp = pd.merge(temp, std, on=[categorical_col])

                #weighted means
                temp['_u'] = ((self._v_prior * self._u_0_prior) + (temp[categorical_col+"_full_count"] * temp[categorical_col+"_sample_mean"])) / \
                    (self._v_prior + temp[categorical_col+"_full_count"])

                #new count
                temp['_v'] = self._v_prior + temp[categorical_col+"_full_count"]

                # new alpha
                temp['_alpha'] = self._alpha_prior + (temp[categorical_col+"_full_count"] / 2)

                #new beta
                temp['_beta'] = self._beta_prior + (.5 * (temp[categorical_col+"_std"]**2)) + \
                    ((temp[categorical_col+"_full_count"] * self._v_prior) / \
                    (temp[categorical_col+"_full_count"] + self._v_prior)) * \
                    (((temp[categorical_col+"_sample_mean"] - self._u_0_prior)**2) / 2)

                #fill NAs with prior
                temp = pd.merge(ALL_LEVELS, temp, on=categorical_col, how='left')
                temp['_u'] = temp['_u'].fillna(self._u_0_prior)
                temp['_v'] = temp['_v'].fillna(self._v_prior)
                temp['_alpha'] = temp['_alpha'].fillna(self._alpha_prior)
                temp['_beta'] = temp['_beta'].fillna(self._beta_prior)

                if categorical_col not in self._gig_distributions.keys():
                    self._gig_distributions[categorical_col] = temp[[categorical_col, '_u', '_v', '_alpha', '_beta']]
                else:
                    self._gig_distributions[categorical_col][['_u', '_v', '_alpha','_beta']] += temp[['_u', '_v', '_alpha','_beta']]

            # report mean alpha and beta:
            self._gig_distributions[categorical_col]['_u'] = self._gig_distributions[categorical_col]['_u']/self._n_samples
            self._gig_distributions[categorical_col]['_v'] = self._gig_distributions[categorical_col]['_v']/self._n_samples
            self._gig_distributions[categorical_col]['_alpha'] = self._gig_distributions[categorical_col]['_alpha']/self._n_samples
            self._gig_distributions[categorical_col]['_beta'] = self._gig_distributions[categorical_col]['_beta']/self._n_samples
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
            if categorical_col not in self._gig_distributions.keys():
                raise AssertionError("Column "+categorical_col+" not fit by GIGEncoder")

            #add `_alpha` and `_beta` columns vi lookups, impute with prior
            X_temp = X_temp.merge(self._gig_distributions[categorical_col], on=[categorical_col], how='left')

            X_temp['_u'] = X_temp['_u'].fillna(self._u_0_prior)
            X_temp['_v'] = X_temp['_v'].fillna(self._v_prior)
            X_temp['_alpha'] = X_temp['_alpha'].fillna(self._alpha_prior)
            X_temp['_beta'] = X_temp['_beta'].fillna(self._beta_prior)         
            
            #   encode with moments
            if 'm' in moments:
                X_temp[categorical_col+'__M_u'] = X_temp["_u"]

                #check alpha > 1
                if (X_temp['_alpha'] <= 1).any():
                    raise ValueError("'alpha' must be greater than 1")
                X_temp[categorical_col+'__M_v'] = X_temp["_beta"] / (X_temp['_alpha'] - 1)
            
            if 'v' in moments:
                
                X_temp[categorical_col+'__V_u'] = X_temp["_beta"] / ((X_temp['_alpha'] - 1) * X_temp['_v'])

                if (X_temp['_alpha'] <= 2).any():
                    raise ValueError("'alpha' must be greater than 2")
                X_temp[categorical_col+'__V_v'] = (X_temp["_beta"]**2) /\
                    (((X_temp['_alpha'] - 1)**2) * (X_temp['_alpha'] - 2))
            
            #and drop columns
            X_temp = X_temp.drop([categorical_col], axis=1)
            X_temp = X_temp.drop(["_u"], axis=1)
            X_temp = X_temp.drop(["_v"], axis=1)
            X_temp = X_temp.drop(["_alpha"], axis=1)
            X_temp = X_temp.drop(["_beta"], axis=1)

        return X_temp

    def get_string_cols(self, df):
        idx = (df.applymap(type) == str).all(0)
        return df.columns[idx]