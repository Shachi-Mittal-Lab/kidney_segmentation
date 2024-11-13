import numpy as np
# sigmoid normalization
# https://en.wikipedia.org/wiki/Normalization_(image_processing)
def sigmoid_norm(gray_array):
    xmax = np.amax(gray_array)
    xmin = np.amin(gray_array)
    newmax = 255
    newmin = 0
    beta = xmax - xmin
    alpha = 255/2
    eterm = np.exp(-(gray_array-beta)/alpha)
    gray_array = (newmax - newmin) * (1/(1+eterm)) + newmin
    return gray_array