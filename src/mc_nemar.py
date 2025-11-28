import numpy as np
# from statsmodels.stats.contingency_tables import mcnemar

print("McNemar's Test for statistical significance")
# Ground truth labels
y_true   = np.array([0,0,0,0,0,1,1,1,1,1,0,0,0,0,0])
# Predictions from classifiers A and B (same test instances)
y_pred_A = np.array([0,0,1,1,1,1,0,1,1,1,0,0,0,0,0])
y_pred_B = np.array([0,1,0,1,1,1,1,1,0,1,1,1,1,1,1])
                     
                     
def contingency_table(y_true,y_pred_A,y_pred_B):
    # Boolean arrays: was each classifier correct on each sample?
    correct_A = (y_pred_A == y_true)
    correct_B = (y_pred_B == y_true)

    # Contingency counts
    # "~" changes true to false
    # "&" logical AND operator
    a = np.sum( correct_A &  correct_B)  # both correct
    b = np.sum( correct_A & ~correct_B)  # A correct, B wrong
    c = np.sum(~correct_A &  correct_B)  # A wrong, B correct
    d = np.sum(~correct_A & ~correct_B)  # both wrong
    # print(correct_A & correct_B)
    # print(~correct_A & ~correct_B)

    print("a (both correct)        =", a)
    print("b (A correct, B wrong)  =", b)
    print("c (A wrong, B correct)  =", c)
    print("d (both wrong)          =", d)

    table = [[a, b],
             [c, d]]
    return table
    
    
print("========== Manual calculation ========== ")                    
table = contingency_table(y_true,y_pred_A,y_pred_B)

print("Contingency table for McNemar:\n", np.array(table))                    

chi2_value = (abs(table[0][1]-table[1][0])-1)**2/(table[0][1]+table[1][0])
print(f"chi2: {chi2_value}") # 1.77777
from scipy.stats import chi2
df = 1
p_value = chi2.sf(chi2_value, df)
print(f"p-value: {p_value}") # 0.1824


print("========== Using libraries ========== ")                    
# https://rasbt.github.io/mlxtend/user_guide/evaluate/mcnemar/
# pip install mlxtend
from mlxtend.evaluate import mcnemar_table   
from mlxtend.evaluate import mcnemar
table = mcnemar_table(y_target=y_true, y_model1=y_pred_A,  y_model2=y_pred_B)
chi2, p = mcnemar(ary=table, corrected=True)
print('chi-squared:', chi2)
print('p-value:', p)

alpha = 0.05
if p < alpha:
    print("Reject H0: classifiers have significantly different accuracy.")
else:
    print("Fail to reject H0: no significant difference in accuracy.")
