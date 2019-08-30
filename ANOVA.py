def one_way_anova(df, colls_to_test, treatment):
    #Function that performs a one way anover on all specified values for the specified treatment
    #Input: df: dataframe the operation is to be run on
    #colls_to_test: list of strings of collum names for which we are going to perform the anova
    #treament: string with treatment collum name
    #output: dataframe containing pvalue and fstat for all collums
    #The resulting dataframe has the collum names as rows and fstat and pval as collums
    
    #First, split the dataset by the treatment
    dframes_sep = seperated_dataframes(df, treatment)
    #Init an empty dictionary which will contain the pvalues
    pv_dict = {}
    #Init empty dictionary which will contain f statistics
    fstat_dict = {}
    #For every collum we want to run our anova on
    for collum in colls_to_test:
        #init empty array which is going to hold the colllum of interest from the seperated dataframes
        seperated_colls = []
        #for each of our previously seperated dataframes
        for df in dframes_sep:
            #obtain the collum of interest less all empty cells and add it to the list of all collums of interest
            seperated_colls.append(df.dropna()[collum])
        #run the one way anova for the sperated collums
        #usually the function f_oneway takes the different collums comma seperated
        #Like f_oneway(coll1, coll2, coll3)
        #The star tells it to treat the list we are passing as such comma seprated variables
        fstat, pval = scs.f_oneway(*seperated_colls)
        #save p value and fstat for this collum to the dict
        pv_dict[collum]=pval
        fstat_dict[collum] = fstat
    #Now we are going to turn those dictionaries into a dataframe
    #We want the collums in the dataframe to have usefull names
    #the convention we use is treatment_fstat or treatment_pval
    #that way we know what we have in there when we look at the dataframe later
    #first we generate the name of the fstat collum by connecting the treatment name with the string'_fstat'
    fstat_coll_name = treatment+'_fstat'
    #then we convert the fstat dict into a dataframe
    fstatdf = pd.DataFrame.from_dict(fstat_dict, orient='index')
    #and then we change the collum name in that dataframe
    fstatdf.columns = [fstat_coll_name]
    #same goes for pvals
    pval_coll_name = treatment+'_pval'
    pvdf = pd.DataFrame.from_dict(pv_dict, orient='index')
    pvdf.columns = [pval_coll_name]
    #then we connect the pval dataframe and the fstat dataframe
    result_dataframe = pd.concat([pvdf,fstatdf],axis=1)
    #and return the result
    return result_dataframe

interesting_collums = ['loyalty', 'satisfaction','educ', 'gender', 'age', 'Involvement', 'Emotional',
       'Calculative', 'Trust', 'Image']
res = one_way_anova(data, interesting_collums, 'loyalty')


all_data_frames = []
for collum in interesting_collums:
    res = one_way_anova(data, interesting_collums, collum)
    all_data_frames.append(res)

final = pd.concat(all_data_frames, axis=1)
final.to_excel('/Users/jannes/Google Drive/ABM/data/all_oneway_anova.xls')



########################################################
# Two way anova
# TODO: Extend for unbalanced design
# I could not find a good module implementation so this is an implementation from scratch
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
