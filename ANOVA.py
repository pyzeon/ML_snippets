''' Seperates dataframe into multiple by treatment
E.g. if treatment is 'gender' with possible values 1 (male) or 2 (female) 
the function returns a list of two frames (one with all males the other with all females) '''

def seperated_dataframes(df, treatment):
    treat_col = data[treatment] # col with the treatment
    dframes_sep = [] # list to hold seperated dataframes 
    for cat in categories(treat_col): # Go through all categories of the treatment
         for the treatmet into a new dataframe
        df = data[treat_col == cat] # select all rows that match the category        
        dframes_sep.append(df) # append the selected dataframe
    return dframes_sep

import numpy as np
import scipy.stats as scs
import pandas as pd

def one_way_anova(df, colls_to_test, treatment):
    dframes_sep = seperated_dataframes(df, treatment) # split the dataset by the treatment
    pv_dict = {}
    fstat_dict = {}
    for collum in colls_to_test: #For every col we want to run our anova on
        seperated_colls = []        
        for df in dframes_sep: # for each of our previously seperated dataframes ...
            seperated_colls.append(df.dropna()[collum]) # ... obtain the col of interest less all empty cells and add it to the list  
        fstat, pval = scs.f_oneway(*seperated_colls) # * tells it to treat the list we are passing as such csv
        pv_dict[collum]=pval
        fstat_dict[collum] = fstat

    fstat_coll_name = treatment+'_fstat'
    fstatdf = pd.DataFrame.from_dict(fstat_dict, orient='index')
    fstatdf.columns = [fstat_coll_name]

    pval_coll_name = treatment+'_pval'
    pvdf = pd.DataFrame.from_dict(pv_dict, orient='index')
    pvdf.columns = [pval_coll_name]
    
    result_dataframe = pd.concat([pvdf,fstatdf],axis=1)
    return result_dataframe

interesting_collums = ['loyalty', 'satisfaction','educ', 'gender', 'age']
res = one_way_anova(data, interesting_collums, 'loyalty')

all_data_frames = []
for collum in interesting_collums:
    res = one_way_anova(data, interesting_collums, collum)
    all_data_frames.append(res)

final = pd.concat(all_data_frames, axis=1)
final.to_excel('/Users/.../data/all_oneway_anova.xls')



########################################################
# Two way anova
# Via: http://www.marsja.se/three-ways-to-carry-out-2-way-anova-with-python/
# Still not perfect so handle with care


data = data.dropna()
#treat1: supp treat2: dose
def two_way_anova(data, measurement, treat1, treat2):
    #Counting all measurements
    N = len(data[measurement])
    a = len(data[treat1].unique())
    b = len(data[treat2].unique())
    r = N/(a*b)
    print(a,b,r)
    #df_a = a - 1 (a # categories a )
    df_a = a - 1
    #df_b = b-1 (b # categories b)
    df_b = b - 1
    # df_axb = (a-1)*(b-1) = df_a*df_b
    df_axb = df_a*df_b
    #df_w (for MSE) = N-a*b (a,b #categories a, b)
    df_w = N - (len(data[treat1].unique())*len(data[treat2].unique()))
    #grand mean = mean for all measurements
    grand_mean = data[measurement].mean()
    
    #The mean of measurement where treat1 is l less the grand mean for all l
    ssq_a = sum([(data[data[treat1] ==l][measurement].mean()-grand_mean)**2 for l in data[treat1]])
    
    ssq_b = sum([(data[data[treat2] ==l][measurement].mean()-grand_mean)**2 for l in data[treat2]])
    print(len(data[measurement]), grand_mean)
    ssq_t = sum((data[measurement] - grand_mean)**2)
    ssq_w = 0
    for cat in data[treat1].unique():
        vc = data[data[treat1] == cat]
        vc_dose_means = [vc[vc[treat2] == d][measurement].mean() for d in vc[treat2]]
        ssq_w += sum((vc[measurement] - vc_dose_means)**2)
       
    ssq_axb = ssq_t-ssq_a-ssq_b-ssq_w
    ms_a = ssq_a/df_a
    ms_b = ssq_b/df_b
    ms_axb = ssq_axb/df_axb
    ms_w = ssq_w/df_w
    f_a = ms_a/ms_w
    f_b = ms_b/ms_w
    f_axb = ms_axb/ms_w
    p_a = scs.f.sf(f_a, df_a, df_w)
    p_b = scs.f.sf(f_b, df_b, df_w)
    p_axb = scs.f.sf(f_axb, df_axb, df_w)
    results = {'sum_sq':[ssq_a, ssq_b, ssq_axb, ssq_w],
           'df':[df_a, df_b, df_axb, df_w],
           'F':[f_a, f_b, f_axb, 'NaN'],
            'PR(>F)':[p_a, p_b, p_axb, 'NaN']}
    columns=['sum_sq', 'df', 'F', 'PR(>F)']

    aov_table1 = pd.DataFrame(results, columns=columns,
                              index=[treat1, treat2, 
                              treat1+':'+treat2, 'Residual'])
    return aov_table1

two_way_anova(data,'loyalty','Trust','Emotional')    
