import pandas as pd
import numpy as np
import seaborn
from matplotlib import pyplot as plt

from multiprocessing import Process, Manager, Pool, cpu_count

import sys
sys.path.append('./../common_utils')
import gen_weather_data
import training
import params

# Import metrics
from sklearn.metrics import f1_score

# Import OP Solvers
from pulp import LpProblem, LpStatus, lpSum, LpVariable, LpMinimize


# Data params
N = params.N
train_perc = params.train_perc
noise_factor = params.noise_factor

# Optimization Problem params
c11 = params.c11
c12 = params.c12
c21 = params.c21
c22 = params.c22


def cost_function(y1, y2, z1, z2):
    cost_rain = z1*(c11 - c12*y1) + c12*y1
    cost_temp = z2*(c21 - c22*y2) + c22*y2
    return cost_rain + cost_temp
    
def cost_function_list(y1_list, y2_list, z1_list, z2_list):
    cost_list = []
    for y1, y2, z1, z2 in zip(y1_list, y2_list, z1_list, z2_list):
        cost_list.append(cost_function(y1, y2, z1, z2))
    return cost_list

def SolOpt(y1, y2, i):
    
    y1 = int(y1)
    y2 = int(y2)

    t_Model = LpProblem(name="weather-problem", sense=LpMinimize)
    z1 = LpVariable(name="z1", cat='Binary')
    z2 = LpVariable(name="z2", cat='Binary')
    
    t_Model+=(z1+z2<=1,"cstr1")
    
    obj_func = cost_function(y1, y2, z1, z2)
    t_Model += obj_func
    status = t_Model.solve()
    var=t_Model.variables()
    
    z1opt = var[0].value() 
    z2opt = var[1].value() 

    return i, z1opt, z2opt


def run_solver(data_test, model_type, results):
    
    
    def collect_result(result):
        results.append(result)
    
    y1_col = 'y1'
    y2_col = 'y2'
    z1_col = 'z1_opt_from_y'  
    z2_col = 'z2_opt_from_y'  
    if model_type != 'real':
        y1_col = 'y1_pred_{}'.format(model_type)
        y2_col = 'y2_pred_{}'.format(model_type)
        z1_col = 'z1_opt_from_y_pred_{}'.format(model_type)
        z2_col = 'z2_opt_from_y_pred_{}'.format(model_type)


    pool = Pool(cpu_count())
    for i in range(0, len(data_test)):
        y1, y2 = data_test[y1_col].iloc[i], data_test[y2_col].iloc[i]
        pool.apply_async(SolOpt, args=(y1, y2, i), callback=collect_result)
    pool.close()
    pool.join()
    
    df_solver = pd.DataFrame(
        data = results, columns = ['ind', z1_col, z2_col]
                ).set_index('ind')

    data_test = pd.concat([data_test, df_solver], axis = 1)
    
    return data_test



def main():

    #########################################################################
    ##### Generate data for lemonade problem ################################
    #########################################################################

    data_weather = gen_weather_data.generate_weather_dataset(
        N = N, noise_factor = noise_factor)

    Ntr = int(train_perc*N)
    Nte = N - Ntr
    print('Data size: Training, Test and Total')
    print('Data training size:', Ntr)
    print('Data test size:    ', Nte)
    print('Total size:        ', N)

    print('\nNoise factor: ',noise_factor)
    suffix_noise = '_noise_' + str(noise_factor).zfill(2).replace('.','')

    data_train = data_weather.iloc[:Ntr, :].reset_index(drop=True).copy()
    data_test = data_weather.iloc[Ntr:, :].reset_index(drop=True).copy()

    feat_cols = ['x1','x2','x3','x4']
    target_col = ['y1','y2']


    #########################################################################
    ##### Set hyperparams for models ########################################
    #########################################################################

    # Hyperparams for SVM
    params_svm = {
        'C':1000
    }


    # Hyperparams for LGB
    params_lgb = {
        'objective':'binary',
        'boosting_type': 'gbdt',
        'learning_rate': 0.02,
        'max_depth':5,
        'num_leaves':7,
        'verbosity':1 
    }



    #########################################################################
    ##### Train: LogReg, SVM and LGB regressors #############################
    #########################################################################

    mdl_log = training.train_logistic(
        X_tr=data_train[feat_cols], y_tr=data_train[target_col])

    mdl_svm = training.train_svm_class(
        X_tr=data_train[feat_cols], y_tr=data_train[target_col], 
        params=params_svm)

    mdl_lgb_1 = training.train_lgb(
        X_tr=data_train[feat_cols], y_tr=data_train[target_col[0]], 
        params=params_lgb)

    mdl_lgb_2 = training.train_lgb(
        X_tr=data_train[feat_cols], y_tr=data_train[target_col[1]], 
        params=params_lgb)



    #########################################################################
    ##### Predict test data: linear, SVM and LGB regressors #################
    #########################################################################

    data_test.loc[:,['y1_pred_lin','y2_pred_lin']] = mdl_log.predict(data_test[feat_cols])
    data_test.loc[:,['y1_pred_svm','y2_pred_svm']] = mdl_svm.predict(data_test[feat_cols])
    data_test.loc[:,'y1_pred_gbm'] = mdl_lgb_1.predict(data_test[feat_cols])
    data_test.loc[:,'y1_pred_gbm'] = np.where(data_test.loc[:,'y1_pred_gbm']>0.5, 1, 0)
    data_test.loc[:,'y2_pred_gbm'] = mdl_lgb_2.predict(data_test[feat_cols])
    data_test.loc[:,'y2_pred_gbm'] = np.where(data_test.loc[:,'y2_pred_gbm']>0.5, 1, 0)


    f1_lin = (f1_score(data_test['y1'], data_test['y1_pred_lin']),
              f1_score(data_test['y2'], data_test['y2_pred_lin']))

    f1_svm = (f1_score(data_test['y1'], data_test['y1_pred_svm']),
              f1_score(data_test['y2'], data_test['y2_pred_svm']))

    f1_lgb = (f1_score(data_test['y1'], data_test['y1_pred_gbm']),
              f1_score(data_test['y2'], data_test['y2_pred_gbm']))

    print('----------Results MSE ------------------')
    print('F1 using linear classifier:', f1_lin)
    print('F1 using SVM classifier:', f1_svm)
    print('F1 using LGBM classifier:', f1_lgb)


    #########################################################################
    ##### Run solver and compute cost functions based on decisions ##########
    #########################################################################


    print('\nSeparated approach: Math solver using real and predicted Y')
    mdl_types = ['lin','svm','gbm','real']
    for mdl_type in mdl_types:
        results = []
        mdl_str = mdl_type
        if mdl_type != 'real':
            mdl_str = mdl_str + ' y predictions'
        else:
            mdl_str = mdl_str + ' y'

        print('Run solver using', mdl_str)
        data_test = run_solver(data_test, model_type=mdl_type, results=results)


    mdl_types = ['real','lin','svm','gbm']
    for mdl_type in mdl_types:
        y1_col = 'y1'
        y2_col = 'y2'
        z1_col = 'z1_opt_from_y'
        z2_col = 'z2_opt_from_y'
        f_col = 'f_opt_from_y'
        if mdl_type!='real':
            z1_col = 'z1_opt_from_y_pred_{}'.format(mdl_type)
            z2_col = 'z2_opt_from_y_pred_{}'.format(mdl_type)
            f_col = 'f_opt_from_y_pred_{}'.format(mdl_type)

        data_test.loc[:, f_col] = cost_function_list(
            y1_list = data_test.loc[:,y1_col],
            y2_list = data_test.loc[:,y2_col],
            z1_list = data_test.loc[:,z1_col],
            z2_list = data_test.loc[:,z2_col])


    #########################################################################
    ##### Run solver and compute cost functions based on decisions ##########
    #########################################################################    

    fobj_cols =   ['f_opt_from_y_pred_lin',
                   'f_opt_from_y_pred_svm',
                   'f_opt_from_y_pred_gbm',
                   'f_opt_from_y']  

    fig, ax = plt.subplots(figsize=(9, 5), dpi=120)
    fig.suptitle('OBJECTIVE FUNCTION DISTRIBUTION x MODEL COMPLEXITY')
    ax = seaborn.violinplot(
        data=data_test[fobj_cols])

    ax.set_xticklabels(['Linear','SVM','LGBM','Real Y'])
    ax.set_xlabel('Model for predictions')
    ax.set_ylabel('Objective Function f')
    fig.savefig('fig_weather_result' + suffix_noise +'.png')

    df_result = pd.concat([
                data_test[fobj_cols].mean(),
                data_test[fobj_cols].median()], axis=1)
    df_result.columns = ['average_cost','median_cost']

    print('----------Results Obj Function----------')
    print(df_result)
    df_result.to_csv(
        'weather_results' + suffix_noise +'.csv', index=False)
    
    
    
if __name__ == "__main__":
    main()