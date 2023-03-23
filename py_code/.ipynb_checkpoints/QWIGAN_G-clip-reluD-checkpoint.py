 
 
#import tensorflow as tf
 #tf.disable_v2_behavior()
# https://github.com/Mohammad-Rahmdel/WassersteinGAN-GradientPenalty-Tensorflow/blob/master/WGAN-GP_MNIST.ipynb
 
def QWIGAN_G_clip  (data_x, data_99, data_m,  gain_parameters):
    
    colmeans = np.nanmean(data_x, axis = 0)
    p = len(colmeans)
    n = np.shape(data_x)[0]
    fillin_mat = np.zeros((n,1),dtype=colmeans.dtype)   #+ colmeans NO COLMEANS!

    batch_size = gain_parameters['batch_size']
    ITER = gain_parameters['iterations']
    C =  gain_parameters['C']
    d = 1e-8
    no, dim = data_x.shape
    h_dim = int(dim)

    data_0 =  np.nan_to_num(data_x, 0) 
    # Normalization2
    norm_data, norm_parameters = normalization(data_x)
    norm_data_x =  np.nan_to_num(norm_data, 0) 
    min = np.nanmin(norm_data_x, axis = 0)
    med = np.nanmean(norm_data_x, axis = 0)
    max = np.nanmax(norm_data_x, axis = 0)

    ## GAIN architecture   
    X = tf.placeholder(tf.float32, shape = [None, dim])
    M = tf.placeholder(tf.float32, shape = [None, dim])
    H = tf.placeholder(tf.float32, shape = [None, dim])

    # Discriminator variables
    D_W1 = tf.Variable(xavier_init([dim, h_dim])) # Data + Hint as inputs
    D_b1 = tf.Variable(tf.zeros(shape = [h_dim]))  
    D_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
    D_b2 = tf.Variable(tf.zeros(shape = [h_dim]))
    D_W3 = tf.Variable(xavier_init([h_dim, dim]))
    D_b3 = tf.Variable(tf.zeros(shape = [dim]))  # Multi-variate outputs
    theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]

    #Generator variables
    # Data + Mask as inputs (Random noise is in missing components)
    G_W1 = tf.Variable(xavier_init([dim, h_dim]))  
    G_b1 = tf.Variable(tf.zeros(shape = [h_dim]))
    G_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
    G_b2 = tf.Variable(tf.zeros(shape = [h_dim]))  
    G_W3 = tf.Variable(xavier_init([h_dim, dim]))
    G_b3 = tf.Variable(tf.zeros(shape = [dim]))
    theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]

    
    def logit(p):
        """element-wise logit of tensor p"""
        return tf.log(tf.div(p + 1e-6 ,1-p + 1e-6) )
        #return - tf.log(1. / (x + 1e-8) - 1. + 1e-6)
        
    ## GAIN functions
    def generator(x):
        # Concatenate Mask and Data
        inputs = x
        G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
        G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)   
        # MinMax normalized output
        G_prob = tf.nn.elu (tf.matmul(G_h2, G_W3) + G_b3) 
        return G_prob

    def discriminator(x):
        # Concatenate Data and Hint
        inputs = x
        D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)  
        D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
        D_logit = tf.matmul(D_h2, D_W3) + D_b3
        D_prob = tf.nn.relu(D_logit)
        return D_prob


    def gradient_penalty(x, x_fake):
        if x.shape[0]!=x_fake.shape[0]:
            raise ValueError('x and x_fake must have the same batch size')     
        temp_shape = x[0].shape
        epsilon = tf.random.uniform(temp_shape, 0.0, 1.0)
        x_hat = epsilon * x + (1 - epsilon) * x_fake
        with tf.GradientTape() as t:
            t.watch(x_hat)
            d_hat = discriminator(x_hat)
        gradients = t.gradient(d_hat, x_hat)
        g_norm2 = tf.sqrt(tf.reduce_sum(gradients ** 2) +1e-8 )
        d_regularizer = tf.reduce_mean((g_norm2 - 1.0) ** 2)
        return d_regularizer    


    Med_loss_record = np.zeros(ITER)
    D_loss_record = np.zeros(ITER) 
    G_loss_record = np.zeros(ITER) 
    G_totloss_record = np.zeros(ITER) 

    p_range =   tf.reshape( tf.range(dim) , [dim])
    G_sample = generator(X)    
    Hat_X = X * M + G_sample * (1-M)

    D_real = discriminator(X)
    D_fake =  discriminator(G_sample)    
    D_hat =  discriminator(Hat_X)    
    pfake = tf.abs(M - D_fake) 
    preal = tf.abs(M - D_real) 

    D_loss_temp= tf.reduce_mean(  ( pfake)**2 ) -tf.reduce_mean(  (preal)**2 ) 
   # D_loss_temp= tf.reduce_mean( logit( pfake)**2 ) -tf.reduce_mean( logit(preal)**2 ) 
    
    X_regularizer = gradient_penalty( X, G_sample)   
    D_loss =  D_loss_temp + C* X_regularizer

    dQ_ob   =   tf.map_fn(lambda j: ob_adj_qtl((X, D_hat, M), j), p_range , fn_output_signature=tf.float32) 
    dQ_msg  =   tf.map_fn(lambda j: msg_adj_qtl( (G_sample, 1-D_hat, M), j), p_range , fn_output_signature=tf.float32) 

    OMDiff =  (  dQ_msg -dQ_ob ) **2
    Med_loss  = tf.reduce_mean(OMDiff)   
    clip_G = [p.assign(tf.clip_by_value(p, 1e-6,  tf.constant(np.inf))) for p in theta_G]
    
    D_solver = tf.train.AdamOptimizer().minimize(-D_loss, var_list=theta_D)
    G_solver = tf.train.AdamOptimizer().minimize(Med_loss, var_list=theta_G)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for it in tqdm(range(1,ITER)):    
        # Sample batch
        batch_idx = sample_batch_index(no, batch_size)
        X_mb = norm_data_x[batch_idx, :]  
        M_mb = data_m[batch_idx, :]  
        
        Z_mb = np.random.uniform(min, max, size = [batch_size, dim])       

        M_colSums = np.sum(M_mb, axis = 0)
        whole_indices = np.where(M_colSums == batch_size) 

        for j in whole_indices[0]: # if there are columns with NO missing 
            batch_noise_ind = sample_batch_index( batch_size , 3)  # just put ONE?? Maybe 2-3?
            M_mb[batch_noise_ind, j] = 0       

        X_mb = M_mb * X_mb + (1-M_mb) * Z_mb        
        _, D_loss_curr = sess.run([D_solver, D_loss],  feed_dict = {M: M_mb, X: X_mb})
        _, Med_loss_curr,_ = sess.run([G_solver, Med_loss, clip_G], feed_dict = {X: X_mb, M: M_mb})
        #_, Med_loss_curr = sess.run([G_solver, Med_loss], feed_dict = {X: X_mb, M: M_mb})

        # NOTE i changed D_loss_temp
        Med_loss_record[it] = Med_loss_curr
        D_loss_record[it] = D_loss_curr
      
    imputed_data =  sess.run([G_sample], feed_dict = {X: norm_data_x, M: data_m})[0]    
    norm_impute  = renormalization(imputed_data, norm_parameters) 
 
    X_hat = data_m * data_0 + (1-data_m) * norm_impute 
    est_M_real = sess.run([D_real], feed_dict = {X: norm_data_x, M: data_m})[0]    
    est_M_fake = sess.run([D_fake], feed_dict = {X: norm_data_x, M: data_m})[0]  
    
    return X_hat, Med_loss_record, D_loss_record, data_m,  est_M_real, est_M_fake
