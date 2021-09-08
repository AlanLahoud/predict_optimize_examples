import pandas as pd
import numpy as np
import seaborn
from matplotlib import pyplot as plt

from multiprocessing import Process, Manager, Pool, cpu_count

import sys
sys.path.append('./../common_utils')
import gen_icecreams_data
import training
import params

# Import metrics
from sklearn.metrics import f1_score, mean_squared_error

# Import OP Solvers
from casadi import SX, qpsol, vertcat


# Data params
N = params.N
noise_factor = params.noise_factor
train_perc = params.train_perc
valid_perc_from_train = params.valid_perc_from_train

# Optimization Problem params
p1 = params.p1
p2 = params.p2



def compute_y_from_predictor(x, h, z1, z2):
    y1 = h[0,0]*x[0] + h[0,1]*x[1] + h[0,2]*z1 + h[0,3]*z2
    y2 = h[1,0]*x[0] + h[1,1]*x[1] + h[1,2]*z1 + h[1,3]*z2
    return y1, y2

def cost_function_pred(y1, y2, z1, z2):
    
    ic1 = p1*(1-z1)*y1
    ic2 = p2*(1-z2)*y2
    return -ic1 - ic2

def cost_function_list(y1_list, y2_list, z1_list, z2_list):
    cost_list = []
    for y1, y2, z1, z2 in zip(y1_list, y2_list, z1_list, z2_list):
        cost_list.append(-cost_function_pred(y1, y2, z1, z2))
    return cost_list


def SolOpt(x, h, i):
    
    z1 = SX.sym('z1');
    z2 = SX.sym('z2');

    constr1 = z1
    constr2 = -z1 + 0.5

    constr3 = z2
    constr4 = -z2 + 0.5

    g = vertcat(constr1,constr2,constr3,constr4)
    
    y1_pred, y2_pred = compute_y_from_predictor(x, h, z1, z2) 
    qp = {'x':vertcat(z1, z2), 
          'f':cost_function_pred(y1_pred, y2_pred, z1, z2), 
          'g':g}

    opts = {}
    opts['printLevel'] = 'none'

    S = qpsol('S', 'qpoases', qp, opts)

    r = S(lbg=0)
    z_opt = r['x']

    z1_opt = z_opt.elements()[0]
    z2_opt = z_opt.elements()[1]
    
    return i, z1_opt, z2_opt



def run_solver(data_test, h, model_type, results):
    
    def collect_result(result):
        results.append(result)
    
    x1_col = 'x1'
    x2_col = 'x2'
    y1_col = 'y1'
    y2_col = 'y2'
    z1_col = 'z1_opt'  
    z2_col = 'z2_opt'  
    
    if model_type == 'lin':
        z1_col = 'z1_opt_from_pred'  
        z2_col = 'z2_opt_from_pred'
        pool = Pool(cpu_count())
        for i in range(0, len(data_test)):
            x = data_test[x1_col].iloc[i], data_test[x2_col].iloc[i]
            pool.apply_async(
                SolOpt, args=(x, h, i), callback=collect_result)
        pool.close()
        pool.join()
        
    else:
        pool = Pool(cpu_count())
        for i in range(0, len(data_test)):
            y1, y2 = data_test[y1_col].iloc[i], data_test[y2_col].iloc[i]
            pool.apply_async(
                SolOptReal, args=(y1, y2, i), callback=collect_result)
        pool.close()
        pool.join()
    
    df_solver = pd.DataFrame(
        data = results, columns = ['ind', z1_col, z2_col]
                ).set_index('ind')

    data_test = pd.concat([data_test, df_solver], axis = 1)
    
    return data_test



def main():

    #########################################################################
    ##### Generate data for icecream problem ################################
    #########################################################################

    data_weather = gen_icecreams_data.generate_icecream_dataset(
        N = N, noise_factor = noise_factor)

    Ntr = int(train_perc*N)
    Nval = int(valid_perc_from_train*Ntr)
    Nte = N - Ntr
    Ntr = Ntr - Nval

    print('Data size: Training, Test and Total')
    print('Data training size:', Ntr)
    print('Data validation size:', Nval)
    print('Data test size:    ', Nte)
    print('Total size:        ', N)

    print('\nNoise factor: ',noise_factor)
    suffix_noise = '_noise_' + str(noise_factor).zfill(2).replace('.','')

    data_train = data_weather.iloc[:Ntr, :].reset_index(drop=True).copy()
    data_val = data_weather.iloc[Ntr:Ntr+Nval, :].reset_index(drop=True).copy()
    data_test = data_weather.iloc[Ntr+Nval:, :].reset_index(drop=True).copy()

    # We dont have the info of discounts in the test data
    data_test.loc[:,['v1','v2']] = np.nan

    # We dont have the info of outcomes in the test data
    data_test.loc[:,['y1','y2']] = np.nan

    feat_cols = ['x1','x2','v1','v2']
    target_col = ['y1','y2']


    #########################################################################
    ##### Train: linear  ####################################################
    #########################################################################

    mdl_lin = training.train_linear(
        X_tr=data_train[feat_cols], y_tr=data_train[target_col])

    h = mdl_lin.coef_ 

    #########################################################################
    ##### Run solver and compute cost functions based on decisions ##########
    #########################################################################

    print('\nSeparated approach: Math solver using predicted Y')
    results = []
    print('Run solver using linear y predictions')
    data_test = run_solver(
        data_test, h, model_type='lin', results=results)

    # We apply the discounts found in best decisions
    data_test.loc[:, 'v1'] = data_test['z1_opt_from_pred']
    data_test.loc[:, 'v2'] = data_test['z1_opt_from_pred']

    # We generate real demand after applied discounts
    data_test.loc[:, 
        ['y1','y2']] = gen_icecreams_data.generate_real_demand_icecreams(
        X=np.array(data_test[['x1','x2','v1','v2']]), 
        noise_factor=noise_factor)


    # Applying zero discount
    data_test_zero_discount = data_test.copy()
    data_test_zero_discount.loc[:, 'v1'] = 0.0
    data_test_zero_discount.loc[:, 'v2'] = 0.0

    # We generate real demand after applied discounts
    data_test_zero_discount.loc[:, 
        ['y1','y2']] = gen_icecreams_data.generate_real_demand_icecreams(
        X=np.array(data_test_zero_discount[['x1','x2','v1','v2']]), 
        noise_factor=noise_factor)


    f_opt_decisions = cost_function_list(
                            y1_list = data_test['y1'], 
                            y2_list = data_test['y2'], 
                            z1_list = data_test['v1'], 
                            z2_list = data_test['v2'])

    f_zero_discount = cost_function_list(
                            y1_list = data_test_zero_discount['y1'], 
                            y2_list = data_test_zero_discount['y2'], 
                            z1_list = data_test_zero_discount['v1'], 
                            z2_list = data_test_zero_discount['v2'])

    data_results = pd.DataFrame(
        data = {
            'f_opt_decisions':f_opt_decisions,
            'f_zero_discount':f_zero_discount,
        })


    #########################################################################
    ##### Extract the resuts ################################################
    #########################################################################     

    fobj_cols = ['f_opt_decisions','f_zero_discount']

    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    fig.suptitle('OBJECTIVE FUNCTION DISTRIBUTION x MODEL COMPLEXITY')

    seaborn.violinplot(
        data=data_results[fobj_cols], ax=ax)
    ax.set_xticklabels(
        ['Linear Y','Zero discount'])
    ax.set_xlabel('Model for predictions')
    ax.set_ylabel('Objective Function f')

    fig.savefig('icecream_result' + suffix_noise +'.png')

    df_result = pd.concat([
                data_results[fobj_cols].mean(),
                data_results[fobj_cols].median()], axis=1)
    df_result.columns = ['average_revenue','median_revenue']

    print('----------Results Obj Function----------')
    print(df_result)
    df_result.to_csv(
        'icecream_results' + suffix_noise +'.csv')


if __name__ == "__main__":
    main()