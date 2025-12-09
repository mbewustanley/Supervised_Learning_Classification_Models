import pandas as pd
import numpy as np


cols = ['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym',
         'fM3Long', 'fM3Trans', 'fAlpha', 'fDist']

def get_parameters(columns):
    parameters = []
    
    for col in columns:
        param = input(f"{col}: ")
        parameters.append(param)

    #podel.predict(parameters)

    
    return parameters


parameters = get_parameters(cols)
print(parameters)