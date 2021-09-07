import pandas as pd
import numpy as np
import seaborn
from matplotlib import pyplot as plt
from tqdm import tqdm

from multiprocessing import Process, Manager, Pool, cpu_count

import sys
sys.path.append('./../common_utils')
import gen_lemonade_data
import training
import params

# Import models
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import lightgbm as lgb

# Import metrics
from sklearn.metrics import mean_squared_error

# Import OP Solvers
from pulp import LpProblem, LpStatus, lpSum, LpVariable, LpMinimize



# Data params
N = params.N
train_perc = params.train_perc
noise_factor = params.noise_factor

# Optimization Problem params
c0 = params.c0
c1 = params.c1
c2 = params.c2

def cost_function_1(y, z):
    return z*(c0 - c1) + c1*y

def cost_function_2(y, z):
    return z*(c0 + c2) - c2*y

def cost_function(y, z):
    if y-z >= 0:
        return cost_function_1(y, z)
    else:
        return cost_function_2(y, z)
    
def cost_function_list(y_list, z_list):
    cost_list = []
    for y, z in zip(y_list, z_list):
        cost_list.append(cost_function(y, z))
    return cost_list
        
def SolOpt_1(y):
    t_Model = LpProblem(name="small-problem", sense=LpMinimize)
    z = LpVariable(name="z", lowBound=0)

    t_Model+=(y-z>=0,"cstr1")
    t_Model+=(z>=0,"cstr2")
    
    obj_func = cost_function_1(y, z)
    t_Model += obj_func
    status = t_Model.solve()
    var=t_Model.variables()
    return var[0].value(),t_Model.objective.value()


def SolOpt_2(y):
    t_Model = LpProblem(name="small-problem", sense=LpMinimize)
    z = LpVariable(name="z", lowBound=0)

    t_Model+=(z-y>=0,"cstr1")
    t_Model+=(z>=0,"cstr2")
    
    obj_func = cost_function_2(y, z)
    t_Model += obj_func
    status = t_Model.solve()
    var=t_Model.variables()
    return var[0].value(),t_Model.objective.value()

def SolOpt(y, i):
    Result_1 = SolOpt_1(y)
    Result_2 = SolOpt_2(y)
    
    if Result_1[1] < Result_2[1]:
        Result = Result_1
    else:
        Result = Result_2
    
    return i, Result[0]
    

def run_solver(data_test, model_type, results):
    
    #pbar = None
    #pbar = tqdm(total=len(data_test))
    
    def collect_result(result):
        results.append(result)
        #pbar.update()
    
    y_col = 'y'
    z_col = 'z_opt_from_y'  
    if model_type != 'real':
        y_col = 'y_pred_{}'.format(model_type)
        z_col = 'z_opt_from_y_pred_{}'.format(model_type)


    pool = Pool(cpu_count())
    for i in range(0, len(data_test)):
        pool.apply_async(SolOpt, args=(data_test[y_col].iloc[i], i), callback=collect_result)
    pool.close()
    pool.join()
    
    df_solver = pd.DataFrame(
        data = results, columns = ['ind', z_col]
                ).set_index('ind')

    data_test = pd.concat([data_test, df_solver], axis = 1)
    
    return data_test




def main():
    
    #########################################################################
    ##### Generate data for lemonade problem ################################
    #########################################################################

    data_lemonade = gen_lemonade_data.generate_lemonade_dataset(
        N = N, noise_factor = noise_factor)

    Ntr = int(train_perc*N)
    Nte = N - Ntr
    print('Data size: Training, Test and Total')
    print('Data training size:', Ntr)
    print('Data test size:    ', Nte)
    print('Total size:        ', N)
    
    print('\nNoise factor: ',noise_factor)
    suffix_noise = '_noise_' + str(noise_factor).zfill(2).replace('.','')

    data_train = data_lemonade.iloc[:Ntr, :].reset_index(drop=True).copy()
    data_test = data_lemonade.iloc[Ntr:, :].reset_index(drop=True).copy()

    feat_cols = ['x1','x2','x3']
    target_col = ['y']



    #########################################################################
    ##### Set hyperparams for models ########################################
    #########################################################################

    # Hyperparams for SVM
    params_svm = {
        'C':1000
    }

    # Hyperparams for LGB
    params_lgb = {
        'objective':'regression',
        'boosting_type': 'gbdt',
        'learning_rate': 0.02,
        'max_depth':5,
        'num_leaves':7,
        'verbosity':-1 
    }



    #########################################################################
    ##### Train: linear, SVM and LGB regressors #############################
    #########################################################################

    mdl_lin = training.train_linear(
        X_tr=data_train[feat_cols], y_tr=data_train[target_col])

    mdl_svm = training.train_svm(
        X_tr=data_train[feat_cols], y_tr=data_train[target_col], 
        params=params_svm)

    mdl_gbm = training.train_lgb(
        X_tr=data_train[feat_cols], y_tr=data_train[target_col], 
        params=params_lgb)



    #########################################################################
    ##### Predict test data: linear, SVM and LGB regressors #################
    #########################################################################

    print('\nSeparated approach: Training process (Linear, SVM and LGBM models)')
    
    data_test.loc[:,'y_pred_lin'] = mdl_lin.predict(data_test[feat_cols])
    data_test.loc[:,'y_pred_svm'] = mdl_svm.predict(data_test[feat_cols])
    data_test.loc[:,'y_pred_gbm'] = mdl_gbm.predict(data_test[feat_cols])

    mse_lin = mean_squared_error(y_true=data_test['y'], 
                                 y_pred=data_test['y_pred_lin'])

    mse_svm = mean_squared_error(y_true=data_test['y'], 
                                 y_pred=data_test['y_pred_svm'])

    mdl_gbm = mean_squared_error(y_true=data_test['y'], 
                                 y_pred=data_test['y_pred_gbm'])

    print('----------Results MSE ------------------')
    print('MSE using linear model:', mse_lin)
    print('MSE using SVM regressor:', mse_svm)
    print('MSE using LGBM regressor:', mdl_gbm)



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
        y_col = 'y'
        z_col = 'z_opt_from_y'
        f_col = 'f_opt_from_y'
        if mdl_type!='real':
            z_col = 'z_opt_from_y_pred_{}'.format(mdl_type)
            f_col = 'f_opt_from_y_pred_{}'.format(mdl_type)

        data_test.loc[:, f_col] = cost_function_list(
            y_list = data_test.loc[:,y_col], 
            z_list = data_test.loc[:,z_col])



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
    fig.savefig('fig_lemonade_result' + suffix_noise +'.png')

    df_result = pd.concat([
                data_test[fobj_cols].mean(),
                data_test[fobj_cols].median()], axis=1)
    df_result.columns = ['average_cost','median_cost']
    df_result.loc[:,'model'] = fobj_cols
    
    print('----------Results Obj Function----------')
    print(df_result)
    df_result.to_csv(
        'lemonade_results' + suffix_noise +'.csv', index=False)

    
if __name__ == "__main__":
    main()

