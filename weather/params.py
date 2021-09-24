#########################################################################
##### Bad Weather problem Parameters# ###################################
#########################################################################

# Data params
N = 6000                    # Total data size (train + test)
train_perc = 0.8            # Percentage of data for training
noise_factor = 0.0          # Add noise to label data (y): try 0 to 0.3

# Optimization Prob params
c11 = 50                     # Cost for the umbrella 
c12 = 100                     # Cost for the rain and ~umbrella
c21 = 40                     # Cost for the coat 
c22 = 200                    # Cost for the cold and ~coat