 
def runMida ( truefile, nafile):
 
#    data_true= np.loadtxt(truefile, delimiter=",")
#    data_99= np.loadtxt(nafile, delimiter=",")
    
     
    data_true = pd.read_csv(truefile, delimiter=",", header=None)
    data_99 = pd.read_csv(nafile, delimiter=",", header=None)     
    data_x = np.where(data_99 < -98, np.NaN, data_99)


    data_x = data_99.replace(-99, np.nan)
    cat_cols_list = []
    imputer = md.Midas(layer_structure = [256,256], vae_layer = False, seed = 89, input_drop = 0.75)
    imputer.build_model(data_x, softmax_columns = cat_cols_list)
    imputer.train_model(training_epochs = 20)
    imputations = imputer.generate_samples(m=1).output_list 
    
    est = np.array(imputations)[0]

    abs_err = abs_error(  est ,  np.array(data_true))
    pct_err = pct_error_na (  est ,np.array(data_99),  np.array(data_true))
    
     
    return abs_err, pct_err, est
