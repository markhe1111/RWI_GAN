'''Utility functions for WRIGAN.

(1) normalization: MinMax Normalizer
(2) renormalization: Recover the data from normalzied data
(3) xavier_init: Xavier initialization
(4) sample_batch_index: sample random batch index
(5)
'''
 
# Necessary packages
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def normalization (data, parameters=None):
  '''Normalize data in [0, 1] range.
  
  Args:
    - data: original data
  
  Returns:
    - norm_data: normalized data
    - norm_parameters: min_val, max_val for each feature for renormalization
  '''

  # Parameters
  _, dim = data.shape
  norm_data = data.copy()
  
  if parameters is None:
  
    # MixMax normalization
    min_val = np.zeros(dim)
    max_val = np.zeros(dim)
    
    # For each dimension
    for i in range(dim):
      min_val[i] = np.nanmin(norm_data[:,i])
      norm_data[:,i] = norm_data[:,i] - np.nanmin(norm_data[:,i])
      max_val[i] = np.nanmax(norm_data[:,i])
      norm_data[:,i] = norm_data[:,i] / (np.nanmax(norm_data[:,i]) + 1e-6)   
      
    # Return norm_parameters for renormalization
    norm_parameters = {'min_val': min_val,
                       'max_val': max_val}

  else:
    min_val = parameters['min_val']
    max_val = parameters['max_val']
    
    # For each dimension
    for i in range(dim):
      norm_data[:,i] = norm_data[:,i] - min_val[i]
      norm_data[:,i] = norm_data[:,i] / (max_val[i] + 1e-6)  
      
    norm_parameters = parameters    
      
  return norm_data, norm_parameters







def renormalization (norm_data, norm_parameters):
  '''Renormalize data from [0, 1] range to the original range.
  
  Args:
    - norm_data: normalized data
    - norm_parameters: min_val, max_val for each feature for renormalization
  
  Returns:
    - renorm_data: renormalized original data
  '''
  
  min_val = norm_parameters['min_val']
  max_val = norm_parameters['max_val']

  _, dim = norm_data.shape
  renorm_data = norm_data.copy()
    
  for i in range(dim):
    renorm_data[:,i] = renorm_data[:,i] * (max_val[i] + 1e-6)   
    renorm_data[:,i] = renorm_data[:,i] + min_val[i]
    
  return renorm_data





 

def xavier_init(size):
  '''Xavier initialization.
  
  Args:
    - size: vector size
    
  Returns:
    - initialized random vector.
  '''
  in_dim = size[0]
  xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
  return tf.random.normal(shape = size, stddev = xavier_stddev)
      


 
def sample_batch_index(total, batch_size):
  '''Sample index of the mini-batch.
  
  Args:
    - total: total number of samples
    - batch_size: batch size
    
  Returns:
    - batch_idx: batch index
  '''
  total_idx = np.random.permutation(total)
  batch_idx = total_idx[:batch_size]
  return batch_idx
  

def qtl_adj (  p_j, X_j , mask ):    

    pi_j = 1/ (1e-8 + p_j )
    w_j = tf.boolean_mask(pi_j, mask)

    wj_sum = tf.reduce_sum( w_j)
    V_j = w_j/ wj_sum
    X_m =    tf.boolean_mask(X_j, mask)
    order_m = tf.argsort(X_m)
    sorted_X = tf.gather(params=X_m , indices=order_m)
    sorted_Vj = tf.gather(params=V_j , indices=order_m)

    adj_Qtl = tf.cumsum(sorted_Vj) 
    XY =  ( tf.concat( [tf.expand_dims(adj_Qtl, 0) ,
                        tf.expand_dims(sorted_X, 0)], axis = 0) )    
    linspace = tf.linspace(0, 1, 41)
    linsp = tf.cast(linspace , dtype = tf.float32)
    interpo = tfp.math.batch_interp_regular_1d_grid(x=linsp,  x_ref_min= [0.], 
                                                    x_ref_max= [1.],  y_ref = XY, axis=-1 )
    return interpo[1]

def ob_adj_qtl(args, j):
    X, estM,M   = args
    X_j = X[:,j]
    M_j = M[:,j]
    mask = M_j >0
    ob_adj   = qtl_adj (estM[:,j], X_j, mask)    
    return ob_adj 

def msg_adj_qtl(args, j):
    X, estN,M   = args
    X_j = X[:,j]
    M_j = M[:,j]
    mask = M_j <1
    msg_adj   = qtl_adj (estN[:,j], X_j, mask)    
    return msg_adj


###-----------------------------------------------------------------------------
