import numpy as np
import pandas as pd

def create_features(N):
    # features X contain:
    # x1: temperature
    # x2: rainfall intensity
    # x3: weekday

    np.random.seed(1)
    x1 = np.random.normal(loc=20.0, scale=4.0, size=N)

    rainfall_intensity = [
        0, # No rain
        1, # Light rain
        2, # Moderate rain 
        3, # Heavy rain
        4  # Violent rain
    ]

    prob_rain = [
        0.70, # No rain prob
        0.15, # Light rain prob
        0.08, # Moderate rain prob
        0.05, # Heavy rain prob
        0.02  # Violent rain prob
    ]

    x2 = np.random.choice(rainfall_intensity, size=N, p=prob_rain)

    weekdays = [
        0, # Sunday
        1, # Monday
        2, # Tuesday
        3, # Wednesday
        4, # Thursday
        5, # Friday
        6, # Saturday
    ]

    x3 = np.random.choice(weekdays, size=N)
    
    X = np.stack((x1, x2, x3), axis=1)
    return X
    

def calculate_real_demand(x):

    # Calculate rain factor
    rain_factor = 1.0
    if x[1] == 4:
        rain_factor = 0.0
    elif x[1] == 3:
        rain_factor = 0.1
    elif x[1] == 2:
        rain_factor = 0.2
    elif x[1] == 1:
        rain_factor = 0.5

    # Calculate weekday factor
    weekdays_factor = 1.0
    if x[2] == 0:
        weekdays_factor = 1.5
    elif x[2] == 6:
        weekdays_factor = 2
    elif x[2] == 5:
        weekdays_factor = 1.5
    elif x[2] == 4:
        weekdays_factor = 1.1
        
    # Generate real demand
    y = ((50 + 10*x[0])*rain_factor*weekdays_factor).clip(min = 0)
    return y

def generate_lemonade_dataset(N):
    X = create_features(N=N)
    y = np.apply_along_axis(func1d = calculate_real_demand, 
                            axis = 1,
                            arr = X)
    data = pd.DataFrame(np.column_stack((X,y)), columns=['x1','x2','x3','y'])
    
    return data