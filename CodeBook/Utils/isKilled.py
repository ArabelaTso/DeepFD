import numpy as np
import pandas as pd
import statsmodels.api as sm

import patsy
import statsmodels.stats.power as pw


# calculates cohen's kappa value
def cohen_d(orig_accuracy_list, accuracy_list):
    nx = len(orig_accuracy_list)
    ny = len(accuracy_list)
    dof = nx + ny - 2
    pooled_std = np.sqrt(
        ((nx - 1) * np.std(orig_accuracy_list, ddof=1) ** 2 + (ny - 1) * np.std(accuracy_list, ddof=1) ** 2) / dof)
    result = (np.mean(orig_accuracy_list) - np.mean(accuracy_list)) / pooled_std
    return result


# calculates whether two accuracy arrays are statistically different according to GLM
def is_diff_sts(orig_accuracy_list, accuracy_list, threshold=0.05):
    len_list = len(orig_accuracy_list)

    zeros_list = [0] * len_list
    ones_list = [1] * len_list
    mod_lists = zeros_list + ones_list
    acc_lists = orig_accuracy_list + accuracy_list

    data = {'Acc': acc_lists, 'Mod': mod_lists}
    df = pd.DataFrame(data)

    response, predictors = patsy.dmatrices("Acc ~ Mod", df, return_type='dataframe')
    glm = sm.GLM(response, predictors)

    # by default, is_kill = 0, meaning that the case is not statistically different from the original distribution
    is_kill = 0
    try:
        glm_results = glm.fit()
    except Exception as e:
        print(e)
        return is_kill

    glm_sum = glm_results.summary()
    pv = str(glm_sum.tables[1][2][4])
    p_value = float(pv)

    effect_size = cohen_d(orig_accuracy_list, accuracy_list)
    is_kill = int((p_value < threshold) and effect_size >= 0.5)
    return is_kill


def power(orig_accuracy_list, mutation_accuracy_list):
    eff_size = cohen_d(orig_accuracy_list, mutation_accuracy_list)
    pow = pw.FTestAnovaPower().solve_power(effect_size=eff_size,
                                           nobs=len(orig_accuracy_list) + len(mutation_accuracy_list), alpha=0.05)
    return pow