 
def runGain ( truefile, nafile,  batch_size, alpha, iterations):
 
    data_true= np.loadtxt(truefile, delimiter=",")
    data_99= np.loadtxt(nafile, delimiter=",")
    
    data_x = np.where(data_99 < -98, np.NaN, data_99)

    gain_parameters = {'batch_size': batch_size,
                         'hint_rate': .5,
                         'alpha': alpha,
                         'iterations': iterations}
    gained = gain(data_x, gain_parameters)
 
    
    abs_err = abs_error(  gained ,  data_true)
    
    pct_err = pct_error_na (  gained ,data_99,  data_true)
    
     
    return abs_err, pct_err, gained
