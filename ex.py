import math
import numpy as np
sig_x= 0.3
sig_y= 0.3
x_obs= 6
y_obs= 3
mu_x= 5
mu_y= 3
# calculate normalization term
gauss_norm= (1/(2 * np.pi * sig_x * sig_y))
# calculate exponent
exponent= ((x_obs - mu_x)**2)/(2 * sig_x**2) + ((y_obs - mu_y)**2)/(2 * sig_y**2)

# calculate weight using normalization terms and exponent
weight= gauss_norm * math.exp(-exponent)
print(weight)
