 
def runRWI ( qwifun, truefile, nafile,  batch_size, C, iterations):
    
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    import tensorflow.compat.v1 as tf
    from tqdm import tqdm
    
    X_true= np.loadtxt(truefile, delimiter=",")
    data_99= np.loadtxt(nafile, delimiter=",")
    
    data_x = np.where(data_99 < -98, np.NaN, data_99)
    n, p = data_x.shape
    data_m = 1-np.isnan(data_x)

    data_true = np.where(X_true < -98, np.NaN, X_true)
    true_m = 1-np.isnan(data_true)


    result = qwifun (data_x, data_m, 
                     {'batch_size':batch_size, 'C':C,  'iterations':iterations})
    
    return result


 
def runWRINN (  truefile, nafile,  batch_size,  iterations):
    
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    import tensorflow.compat.v1 as tf
    from tqdm import tqdm
    
    X_true= np.loadtxt(truefile, delimiter=",")
    data_99= np.loadtxt(nafile, delimiter=",")
    
    data_x = np.where(data_99 < -98, np.NaN, data_99)
    n, p = data_x.shape
    data_m = 1-np.isnan(data_x)

    data_true = np.where(X_true < -98, np.NaN, X_true)
    true_m = 1-np.isnan(data_true)


    result = WRINN (data_x, data_m, 
                     {'batch_size':batch_size,   'iterations':iterations})
    
    return result
