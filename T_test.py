
def t_test(subset1, subset2, collums_to_test):
    #Function which computes the t test for equal means for all collums specified
    #Input: subset1, subset2: DataFrames, split by some ordinal variable
    #Output: result_dataframe: a DataFrame containing p value, t statistic and the result of the f test as collums
    #... and the different variables which where tested as rows
    
    #First, we set up some dictionaries to temorarily hold the values before we move them to the dataframe
    f_test_dict = {}
    t_stat_dict = {}
    p_val_dict = {}
    #For every collum we are supposed to test...
    for collum in collums_to_test:
        #Check whether the collum is present in both subsets (if not just skip it)
        if not collum in subset1 or not collum in subset2:
            continue
        #Measure veriance in first subset
        df1_var = subset1[collum].var()
        #Measure sample size n in first subset
        df1_n = subset1[collum].count()
        #Same for the second subset
        df2_var = subset2[collum].var()
        df2_n = subset2[collum].count()
        #Compute the F statistic
        F = df1_var/df2_var
        #Compute the critical F value
        Fcritical = scs.f.ppf(0.95,df1_n-1,df2_n-1)
        #If F<Fcritical we have not enough evidence to reject H0: Equal variance
        equal_variance = (F<Fcritical)
        #We now compute the t test
        #dropna() just removes all empty cells, which makes the test run smooth
        #If a a single cell would be empty, the t test would just return Na 
        #Note how we use the result of the F test to tell the t test whether variance can be assumed equal 
        t_stat, p_value  = scs.ttest_ind(subset1.dropna()[collum],subset2.dropna()[collum],equal_var=equal_variance)
        #We then save the reuslts in our temporary dictionaries
        f_test_dict[collum]=equal_variance
        t_stat_dict[collum]=t_stat
        p_val_dict[collum]=p_value
    #After we have done all the t tests, we have to transform those dictionaries to a nice dataframe
    #First we turn the dictionary into a dataframe, this gives us all indices of the dictionary as rows
    pvdf = pd.DataFrame.from_dict(p_val_dict, orient='index')
    #since it does not give a collum name, we have to add it here
    pvdf.columns = ['p value']
    tstatdf = pd.DataFrame.from_dict(t_stat_dict, orient='index')
    tstatdf.columns = ['t stat']
    ftdf = pd.DataFrame.from_dict(f_test_dict, orient='index')
    ftdf.columns = ['Equal Variance']
    #Now we concentate all three data frames
    result_dataframe = pd.concat([pvdf,tstatdf,ftdf],axis=1)
    #and return the whole thing
    return result_dataframe

#usage
#Split data into subsets by gender
males = data[data['gender']==1]
#remove the gender collum form the subsets (since it is useless in the t test)
del males['gender']
females = data[data['gender']==2]
del females['gender']



#run function
result_dataframe = t_test(males,females,colls_to_test)

#save dataframe as excel file
result_dataframe.to_excel('/Users/jannes/Google Drive/ABM/data/t_test_by_gender.xls')
