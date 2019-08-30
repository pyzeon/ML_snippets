#Chi square function which performs an independence test between two
#collums of a data frame
def chi_square_of_df_cols(df, col1_name, col2_name):
    #extract the actual collums from the data frame
    df_col1, df_col2 = df[col1_name], df[col2_name]
    
    #the result matrix is a nested array of observed frequencies
    #visualize it as every list being a collum in the observed frequencies
    #and the whole nested list being a list of all collums
    #we just set this up to be empty here
    result = []
    #In the outer loop we go over every category for the first variable
    for cat1 in categories(df_col1):
        #we set up the inner list (the collums)
        cat1_value_list = []
        #now we loop over every category in the second variable
        for cat2 in categories(df_col2):
            #we count the number of occurences of cat1 and cat2 in the data
            num_measurements = len(df[ (df_col1 == cat1) & (df_col2 == cat2) ])
            #and append that value to the list
            cat1_value_list.append(num_measurements)
        #then we append the inner list (collum) to the overall nested list
        result.append(cat1_value_list)
    #now we let scipy stats run the chi square test
    chi2, p_value, df, expected = scs.chi2_contingency(result)
    #and return chi2, p value and degrees of freedom
    return chi2, p_value, df
    

chi_square_of_df_cols(data, 'loyalty', 'educ')

#Lets perform the chi2 test for every collum against every other
#we are not interested in all collums, just those 10
interesting_collums = ['loyalty', 'satisfaction','educ', 'gender', 'age', 'Involvement', 'Emotional',
       'Calculative', 'Trust', 'Image']
n = len(interesting_collums)


#we save the resulting p values in a numpy matrix
import numpy as np
pval_mat = np.empty((n,n))

#Go to the collums
for i,name1 in enumerate(interesting_collums):
    #and for every collum go through all other collums
    for j, name2 in enumerate(interesting_collums):
        #perform the chi2 of those two
        chi2, p_value, df = chi_square_of_df_cols(data, name1, name2)
        #and just save the p value (we could use the other ones but nah)
        pval_mat[i,j] = p_value

#Save all the p vals as a text file which you can load e.g. in excel
pval_df = pd.DataFrame(pval_mat, columns=interesting_collums, index=interesting_collums)
pval_df.to_excel('/Users/jannes/Google Drive/ABM/data/all_chi2_pvals.xls')
np.savetxt('/Users/jannes/Google Drive/ABM/data/chi2_pvals.csv',pval_mat)

sns.heatmap(pval_df, square = True)
