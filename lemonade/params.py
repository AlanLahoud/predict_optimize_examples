#########################################################################
##### Lemonade Stand problem Parameters# ################################
#########################################################################

# Data params
N = 6000                    # Total data size (train + test)
train_perc = 0.8            # Percentage of data for training
noise_factor = 0.2          # Add noise to label data (y): try 0 to 0.3

# Optimization Prob params
c0 = 10                     # Constant for the order cost
c1 = 50                     # Constant for the shortage cost
c2 = 30                     # Constant for the waste cost
pen = 200                     # Constant for the waste cost