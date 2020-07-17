# used in LDA

import numpy as np

def data_to_dict(X, Y):
    data_dict = {}
    for x, y in zip(X, Y):
        if y not in data_dict:
            data_dict[y] = [x.flatten()]
        else:
            data_dict[y].append(x.flatten())
            
    for key in data_dict:
        data_dict[key] = np.asarray(data_dict[key])
        
    return data_dict