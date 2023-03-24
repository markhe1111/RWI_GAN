 

 
def runGRWINN (   nafile,  batch_size,  iterations):
    
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    import tensorflow.compat.v1 as tf
    from tqdm import tqdm
    
    data_99= np.loadtxt(nafile, delimiter=",")
    
    data_x = np.where(data_99 < -98, np.NaN, data_99)
    n, p = data_x.shape
    data_m = 1-np.isnan(data_x)
 

    result = GRWINN (data_x, data_m, 
                     {'batch_size':batch_size,   'iterations':iterations})
    
    return result
