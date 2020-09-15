# t test for equal means for all collums specified

import numpy as np
import scipy.stats as scs
import pandas as pd

def t_test(subset1, subset2, collums_to_test):
    f_test_dict = {}
    t_stat_dict = {}
    p_val_dict = {}
    
    for collum in collums_to_test: #For every collum we are supposed to test...        
        if not collum in subset1 or not collum in subset2: # ... check whether the col is present in both subsets
            continue
        
        df1_var = subset1[collum].var() # variance in first subset
        df2_var = subset2[collum].var()
        df1_n = subset1[collum].count() # sample size n in first subset
        df2_n = subset2[collum].count()

        F = df1_var/df2_var # F statistic
        Fcritical = scs.f.ppf(0.95,df1_n-1,df2_n-1) # critical F value
        equal_variance = (F<Fcritical) #If F<Fcritical we have not enough evidence to reject H0: Equal variance
        t_stat, p_value  = scs.ttest_ind(subset1.dropna()[collum],subset2.dropna()[collum], # t test
                                         equal_var=equal_variance) # F test tells the t test whether var can be assumed equal

        f_test_dict[collum]=equal_variance
        t_stat_dict[collum]=t_stat
        p_val_dict[collum]=p_value
        
    pvdf = pd.DataFrame.from_dict(p_val_dict, orient='index')
    pvdf.columns = ['p value']
    tstatdf = pd.DataFrame.from_dict(t_stat_dict, orient='index')
    tstatdf.columns = ['t stat']
    ftdf = pd.DataFrame.from_dict(f_test_dict, orient='index')
    ftdf.columns = ['Equal Variance']
    result_dataframe = pd.concat([pvdf,tstatdf,ftdf],axis=1)
    return result_dataframe

#usage
males = data[data['gender']==1]
del males['gender']
females = data[data['gender']==2]
del females['gender']
result_dataframe = t_test(males,females,colls_to_test)

result_dataframe.to_excel('/Users/.../data/t_test_by_gender.xls')
