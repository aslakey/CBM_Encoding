import pandas as pd
import numpy as np

class DirichletEncoder:
    '''DirichletEncoder

    DirichletEncoder is used to encode categorical features with a Dirichlet-Multinomial conjugate pair
    model (i.e. a Dirichlet posterior predictive distribution for binary target). 
    For each categorical feature, this object stores a vector of P(Li = Cj) -- 
    with Cj being class j and Li being level i -- for each existing level of the categorical feature.  

    The input to fit() should be an array-like of integers for y
    and array-like of strings for X.
    The output of transform() will be <column>__[M]_j where [M] is a particular moment 
    of the Dirichlet distribution [‘mvsk’] (m is default) and j is the class of y.
    By default, a prior of alpha_j = 1/J (uniform) is used.

    Parameters
    ----------
    alpha_priors (dict of floats): prior for dirichlet alpha's. default = 1/J
        where J is the number of classes
    random_state (integer): random state for bootstrap samples. default = 1
    n_samples (integer): number of bootstrap samples. default = 100

    Attributes
    ----------
    _alpha_priors (dict of floats) - prior for dirichlet alpha's. default = 1/J
        where J is the number of classes
    _random_state (integer): random state for bootstrap samples. default = 1
    _n_samples (integer): number of bootstrap samples. default = 100
    _dirichlet_distributions (dict) - houses the dirichlet parameters
        in dictionary


    Methods
    ----------
    fit()
    transform()


    Examples
    --------
    >>>import pandas as pd
    >>>import numpy as np
    >>>import io
    >>>import requests
    >>>from dirichlet_encoder import DirichletEncoder
    >>>url="https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
    >>>r=requests.get(url).content
    >>>df=pd.read_csv(io.StringIO(r.decode('utf-8')),header=None)
    >>>names=['buying',
    ...    'maint',
    ...    'doors',
    ...    'persons',
    ...    'lug_boot',
    ...    'safety',
    ...    'class']
    >>>new_name = dict(enumerate(names))
    >>>df = df.rename(new_name,axis='columns')
    >>>#Creating the dependent variable class
    >>>factor = pd.factorize(df['class'])
    >>>df['class'] = factor[0]
    >>>definitions = factor[1]
    >>>print(df['class'].head())
    >>>print(definitions)
    >>>encoder=DirichletEncoder()
    >>>encoder.fit(df[['buying','maint','doors','persons','lug_boot','safety']],\
    >>>df['class'])
    '''

    def __init__(self, n_samples=10, sample_size=.75, random_state=1, alpha_priors=None):
        '''init for BetaEncoder
        Args:
            alpha - prior for number of successes
            beta - prior for number of failures

        '''
        # Validate Types
        if alpha_priors == None:
            alpha_priors = dict()
        if type(alpha_priors) != dict:
            raise AttributeError("Argument 'alpha_priors' must be of type dict")
        if type(n_samples) is not int:
            raise AttributeError("Argument 'n_samples' must be of type int")
        if type(random_state) is not int:
            raise AttributeError("Argument 'random_state' must be of type int")

        #Assign
        self._alpha_priors = alpha_priors
        self._dirichlet_distributions = dict()
        self._random_state = random_state
        self._n_samples = n_samples
        self._sample_size = .5
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
        #convert y to series:
        if type(y) == pd.DataFrame:
            y = y.ix[:,0]

        #fit alpha_priors
        if len(self._alpha_priors.keys()) == 0:
            #fill with class_j : count_j / count
            temp = y.value_counts().to_dict()
            full_count = len(y)
            self._alpha_priors ={k:temp[k]/full_count for k in temp.keys()}
            #print(self._alpha_priors)
        
        #make sure all classes are in keys:
        if not set(y.unique()).issubset(set(self._alpha_priors.keys())):
            print(set(self._alpha_priors.keys()))
            print("")
            print(set(y.unique()))
            raise AssertionError("All possible classes of 'y' must be in alpha_priors")

        X_temp = X.copy(deep=True)
        categorical_cols = columns
        if not categorical_cols:
            categorical_cols = self.get_string_cols(X_temp)

        #add target
        target_col = '_target'
        X_temp[target_col] = y


        for categorical_col in categorical_cols:

            # All Levels
            # Bootstrap samples may not contain all levels, so fill NA with priors
            ALL_LEVELS = X_temp[[categorical_col, target_col]].groupby(categorical_col).count().reset_index()

            for i in range(self._n_samples):

                X_sample = X_temp[[categorical_col, target_col]].sample(n=int(len(X_temp)*self._sample_size), replace=True, random_state=self._random_state + i)

                #alphas
                alpha_dicts = dict()
                positive_counts = pd.DataFrame()

                for k in self._alpha_priors.keys():
                    
                    #prior for dirichlet distribution
                    prior = self._alpha_priors[k]
                    
                    #column name for this data frame
                    alpha_col = k
                    
                    # Get positive examples
                    temp = X_sample[[categorical_col, target_col]]
                    alpha_k = temp[temp[target_col] == k].groupby(categorical_col).count().reset_index()
                    alpha_k = alpha_k.rename(index=str, columns={target_col: categorical_col+'_'+str(k)+"_positive_count"})
                    
                    #add prior
                    alpha_k[categorical_col+'_'+str(k)+"_positive_count"] = alpha_k[categorical_col+'_'+str(k)+"_positive_count"] + prior

                    #start from all levels and merge in  alpha_k
                    alpha_k = pd.merge(ALL_LEVELS, alpha_k, on=categorical_col, how='left')
                    alpha_k = alpha_k.fillna(prior)
                    
                    # data frame of [level, positive_count]
                    alpha_dicts[alpha_col] = alpha_k[[categorical_col,categorical_col+'_'+str(k)+"_positive_count"]]
                    
                    # now we have one sample of alpha_k + prior_k for each level
                
                #outer dirichlet dictionary
                if categorical_col not in self._dirichlet_distributions.keys():
                    self._dirichlet_distributions[categorical_col] = alpha_dicts
                
                #now fill in inner alpha_k's
                else:
                    for k in alpha_dicts.keys():
                        self._dirichlet_distributions[categorical_col][k][categorical_col+'_'+str(k)+"_positive_count"] += alpha_dicts[k][categorical_col+'_'+str(k)+"_positive_count"] 
            
            #last loop to report mean alphas:
            for k in alpha_dicts.keys():
                self._dirichlet_distributions[categorical_col][k][categorical_col+'_'+str(k)+"_positive_count"] = \
                    self._dirichlet_distributions[categorical_col][k][categorical_col+'_'+str(k)+"_positive_count"] / self._n_samples
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
            if categorical_col not in self._dirichlet_distributions.keys():
                raise AssertionError("Column "+categorical_col+" not fit by BetaEncoder")
                
            # a0_ = sum of all alphas
            X_temp[categorical_col + '__a0'] = 0

            #add `_alpha_k` via lookups, impute with prior
            alphas = self._dirichlet_distributions[categorical_col]
            for k in alphas.keys():

                X_temp = X_temp.merge(alphas[k], on=[categorical_col], how='left')

                X_temp[categorical_col+'_alpha_'+str(k)] = X_temp[categorical_col+'_'+str(k)+"_positive_count"].fillna(self._alpha_priors[k])
                X_temp[categorical_col + '__a0'] += X_temp[categorical_col+'_alpha_'+str(k)]
                
            
            # have to run over again to compute mean and variance
            for k in alphas.keys():

                #   encode with moments
                if 'm' in moments:
                    X_temp[categorical_col+'__M__'+str(k)] = X_temp[categorical_col+'_alpha_'+str(k)]/\
                                X_temp[categorical_col + '__a0']
                
                if 'v' in moments:
                    X_temp[categorical_col+'__V__'+str(k)] = (X_temp[categorical_col+'_alpha_'+str(k)] * \
                                (X_temp[categorical_col + '__a0'] - X_temp[categorical_col+'_alpha_'+str(k)])) / \
                                (((X_temp[categorical_col + '__a0'])**2)*(X_temp[categorical_col + '__a0'] + 1))

                #drop alpha_k and positive count
                X_temp = X_temp.drop([categorical_col+'_alpha_'+str(k)], axis=1)
                X_temp = X_temp.drop([categorical_col+'_'+str(k)+"_positive_count"], axis=1)
            
            #now drop category columns
            X_temp = X_temp.drop([categorical_col], axis=1)
            X_temp = X_temp.drop([categorical_col + '__a0'], axis=1)

        return X_temp

    def get_string_cols(self, df):
        idx = (df.applymap(type) == str).all(0)
        return df.columns[idx]