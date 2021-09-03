import numpy as np
import pandas as pd

def create_features(N):

    np.random.seed(1)

    x1 = np.random.normal(loc=0, scale=1, size=N)
    x2 = np.random.normal(loc=0, scale=1, size=N)
    x3 = np.random.normal(loc=0, scale=1, size=N)
    x4 = np.random.normal(loc=0, scale=1, size=N)
    
    X = np.stack((x1, x2, x3, x4), axis=1)

    return X


def generate_real_weather(X, noise_factor):

    y1 = (X[:,0]**2 + X[:,0]*X[:,1] + X[:,1]*X[:,2]*X[:,3] + X[:,3]**3 - 3)
    noise_y1 = np.random.normal(0, noise_factor*y1.std(), size = len(y1))
    y1 = y1 + noise_y1
    y1 = np.where(y1<=0, 0, 1)

    y2 = (X[:,1]**3 + 3*X[:,2]*X[:,3] + 8*X[:,2]**3 - 3*X[:,1]*X[:,2]**2)
    noise_y2 = np.random.normal(0, noise_factor*y2.std(), size = len(y2))
    y2 = y2 + noise_y2
    y2 = np.where(y2<=0, 0, 1)
    
    Y = np.stack((y1, y2), axis=1)
    
    return Y


def generate_weather_dataset(N, noise_factor):
    X = create_features(N=N) 
    Y = generate_real_weather(X, noise_factor=noise_factor)
    data = pd.DataFrame(np.column_stack((X,Y)), columns=['x1','x2','x3','x4','y1','y2'])
    
    return data