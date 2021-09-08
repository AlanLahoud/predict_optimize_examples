import numpy as np
import pandas as pd

def create_features(N):
    # features X contain:
    # x1: temperature
    # x2: competing stores
    # v1: discount ice cream 1
    # v2: discount ice cream 2

    np.random.seed(1)
    x1 = np.random.normal(loc=20.0, scale=6.0, size=N)
    x2 = np.random.randint(0, 10, size=N)
    
    v1 = (5.0/100.0)*np.random.randint(0, 9, size=N)
    v2 = (5.0/100.0)*np.random.randint(0, 7, size=N)
    
    X = np.stack((x1, x2, v1, v2), axis=1)
    return X  
    

def generate_real_demand_icecreams(X, noise_factor):

    y1 = (100 + 0.5*X[:,0]**2 - 10*X[:,1] + 4000*X[:,2]**2 - 2000*X[:,3]**2)
    noise_y1 = np.random.normal(0, noise_factor*y1.std(), size = len(y1))
    y1 = (y1 + noise_y1).clip(min = 0)

    y2 = (200 + 0.3*X[:,0]**2 - 10*X[:,1] - 2000*X[:,2]**2 + 4000*X[:,3]**2)
    noise_y2 = np.random.normal(0, noise_factor*y2.std(), size = len(y2))
    y2 = (y2 + noise_y2).clip(min = 0)
    
    Y = np.stack((y1, y2), axis=1)
    
    return Y


def generate_icecream_dataset(N, noise_factor):
    X = create_features(N=N) 
    Y = generate_real_demand_icecreams(X, noise_factor=noise_factor)
    data = pd.DataFrame(np.column_stack((X,Y)), 
                        columns=['x1','x2','v1','v2','y1','y2'])
    
    return data