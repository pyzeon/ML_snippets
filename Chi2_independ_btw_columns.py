import numpy as np
import scipy.stats as scs
import pandas as pd

#Chi square function which performs an independence test btw 2 cols of a data frame
def chi_square_of_df_cols(df, col1_name, col2_name):
    df_col1, df_col2 = df[col1_name], df[col2_name]
    result = []
    for cat1 in categories(df_col1): #In the outer loop we go over every category for the first variable
        cat1_value_list = []
        for cat2 in categories(df_col2): #now we loop over every category in the second variable
            num_measurements = len(df[ (df_col1 == cat1) & (df_col2 == cat2) ]) # number of occurences 
            cat1_value_list.append(num_measurements)
        result.append(cat1_value_list)
    chi2, p_value, df, expected = scs.chi2_contingency(result) # scipy stats run the chi square test
    return chi2, p_value, df # chi2, p value and degrees of freedom
    
chi_square_of_df_cols(data, 'loyalty', 'educ')

#Lets perform the chi2 test for every collum against every other
interesting_collums = ['loyalty', 'satisfaction','educ', 'gender', 'age', 'Involvement']
n = len(interesting_collums)
pval_mat = np.empty((n,n))
for i,name1 in enumerate(interesting_collums):
    for j, name2 in enumerate(interesting_collums): # for every collum go through all other collums
        chi2, p_value, df = chi_square_of_df_cols(data, name1, name2) #perform the chi2 of those two
        pval_mat[i,j] = p_value

#Save all the p vals as a text file which you can load e.g. in excel
pval_df = pd.DataFrame(pval_mat, columns=interesting_collums, index=interesting_collums)
pval_df.to_excel('/Users/.../data/all_chi2_pvals.xls')
np.savetxt('/Users/.../data/chi2_pvals.csv',pval_mat)
