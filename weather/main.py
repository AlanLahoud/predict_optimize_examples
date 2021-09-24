import pandas as pd
import numpy as np
import seaborn as sns
sns.set_theme(style="whitegrid")
from matplotlib import pyplot as plt

from multiprocessing import Process, Manager, Pool, cpu_count

import sys
sys.path.append('./../common_utils')
import gen_weather_data
import training
import params

# Import metrics
from sklearn.metrics import f1_score, mean_squared_error

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


def run_solver(data_test, model_type, bool_type, results):
    
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
        if bool_type:
            y1_col = y1_col + '_bool'
            y2_col = y2_col + '_bool'
            z1_col = z1_col + '_bool'
            z2_col = z2_col + '_bool'

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
    ##### Generate data for weather problem #################################
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
        'verbosity':-1 
    }



    #########################################################################
    ##### Train: LogReg, SVM and LGB classifiers ############################
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


    data_test.loc[:,['y1_pred_lin','y2_pred_lin']] = np.array(
        mdl_log.predict_proba(data_test[feat_cols]))[:,:,1].T

    data_test.loc[:,['y1_pred_svm','y2_pred_svm']] = np.array(
        mdl_svm.predict_proba(data_test[feat_cols]))[:,:,1].T

    data_test.loc[:,'y1_pred_lgb'] = mdl_lgb_1.predict(data_test[feat_cols])
    data_test.loc[:,'y2_pred_lgb'] = mdl_lgb_2.predict(data_test[feat_cols])

    preds_col = [c for c in data_test.columns.tolist() if '_pred' in c]
    for c in preds_col:
        data_test.loc[:, c+'_bool'] = np.where(data_test.loc[:,c]>0.5, 1, 0)

    f1_lin = (f1_score(data_test['y1'], data_test['y1_pred_lin_bool']).round(3),
              f1_score(data_test['y2'], data_test['y2_pred_lin_bool']).round(3))

    f1_svm = (f1_score(data_test['y1'], data_test['y1_pred_svm_bool']).round(3),
              f1_score(data_test['y2'], data_test['y2_pred_svm_bool']).round(3))

    f1_lgb = (f1_score(data_test['y1'], data_test['y1_pred_lgb_bool']).round(3),
              f1_score(data_test['y2'], data_test['y2_pred_lgb_bool']).round(3))


    mse_lin = (mean_squared_error(
              y_true=data_test['y1'], y_pred=data_test['y1_pred_lin']).round(3),
               mean_squared_error(
              y_true=data_test['y2'], y_pred=data_test['y2_pred_lin']).round(3))

    mse_svm = (mean_squared_error(
              y_true=data_test['y1'], y_pred=data_test['y1_pred_svm']).round(3),
               mean_squared_error(
              y_true=data_test['y2'], y_pred=data_test['y2_pred_svm']).round(3))

    mse_lgb = (mean_squared_error(
              y_true=data_test['y1'], y_pred=data_test['y1_pred_lgb']).round(3),
               mean_squared_error(
              y_true=data_test['y2'], y_pred=data_test['y2_pred_lgb']).round(3))


    print('--------------------Results F1 -----------------------')
    print('F1 using linear classifier (binary):', 
          '\tRain:', f1_lin[0], '\tCold:', f1_lin[1])
    print('F1 using SVM classifier (binary):', 
          '\tRain:', f1_svm[0], '\tCold:', f1_svm[1])
    print('F1 using LGBM classifier (binary):', 
          '\tRain:', f1_lgb[0], '\tCold:', f1_lgb[1])

    print('--------------------Results Brier Score --------------')
    print('Brier using linear classifier (proba):', 
          '\tRain:', mse_lin[0], '\tCold:', mse_lin[1])
    print('Brier using SVM classifier (proba):', 
          '\tRain:', mse_svm[0], '\tCold:', mse_svm[1])
    print('Brier using LGBM classifier (proba):', 
          '\tRain:', mse_lgb[0], '\tCold:', mse_lgb[1])


    #########################################################################
    ##### Run solver and compute cost functions based on decisions ##########
    #########################################################################

    print('\nSeparated approach: Math solver using real and predicted Y')
    mdl_types = ['lin','svm','lgb','real']
    for mdl_type in mdl_types:

        mdl_str = mdl_type
        if mdl_type != 'real':
            mdl_str = mdl_str + ' y predictions'
            results = []
            print('Run solver using', mdl_str)
            data_test = run_solver(
                data_test, model_type=mdl_type, bool_type=False, results=results)

            results = []
            data_test = run_solver(
                data_test, model_type=mdl_type, bool_type=True, results=results)

        else:
            results = []
            print('Run solver using actual y') 
            data_test = run_solver(
                data_test, model_type=mdl_type, bool_type=False, results=results)



    z1_cols = [c for c in data_test.columns.tolist() if 'z1_opt_from_y' in c]
    z2_cols = [c for c in data_test.columns.tolist() if 'z2_opt_from_y' in c]

    f_cols = [c.replace('z1_','f_') for c in data_test.columns.tolist() if 'z1_opt_from_y' in c]
    f_cols_proba = [c for c in f_cols if 'pred' in c and 'bool' not in c]
    f_cols_bool = [c for c in f_cols if 'pred' in c and 'bool' in c]

    for z1_col, z2_col, f_col in zip(z1_cols, z2_cols, f_cols):
        data_test.loc[:, f_col] = cost_function_list(
            y1_list = data_test.loc[:,'y1'],
            y2_list = data_test.loc[:,'y2'],
            z1_list = data_test.loc[:,z1_col],
            z2_list = data_test.loc[:,z2_col])


    #########################################################################
    ##### Extract the resuts ################################################
    #########################################################################     

    cmodels = 'cornflowerblue'
    ctruth = 'seagreen'

    fig, ax = plt.subplots(figsize=(10, 10), dpi=120)
    sns.violinplot(
        data=data_test[f_cols_bool + ['f_opt_from_y']], ax=ax,
        palette=[cmodels,cmodels,cmodels,ctruth])
    ax.set_xticklabels(
        ['Linear','SVM','LGBM', 'Ground truth'])
    ax.set_xlabel('Model for predictions')
    ax.set_ylabel('Objective Function f')
    ax.set_ylim([-50, 350])
    fig.savefig('fig_weather_result' + suffix_noise +'.png')


    fig, ax = plt.subplots(figsize=(10, 10), dpi=120)
    sns.violinplot(
        data=data_test[f_cols_proba + ['f_opt_from_y']], ax=ax,
        palette=[cmodels,cmodels,cmodels,ctruth])
    ax.set_xticklabels(
        ['Linear','SVM','LGBM', 'Ground truth'])
    ax.set_xlabel('Model for predictions')
    ax.set_ylabel('Objective Function')
    ax.set_ylim([-50, 350])
    fig.savefig('fig_weather_result_proba' + suffix_noise +'.png')

    df_result = pd.concat([
                data_test[f_cols].mean(),
                data_test[f_cols].median()], axis=1)
    df_result.columns = ['average_cost','median_cost']
    df_result.loc[:,'model'] = f_cols

    print('----------Results Obj Function----------')
    print(df_result)
    df_result.to_csv(
        'weather_results' + suffix_noise +'.csv', index=False)
    
    
if __name__ == "__main__":
    main()