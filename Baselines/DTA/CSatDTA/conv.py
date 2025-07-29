from keras.layers import Layer
from keras.layers import Conv2D
from keras.layers import Concatenate, concatenate, Reshape
from keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D



from keras import initializers




def _conv_layer1d(ip, t_n, f_n, filters, kernel_size, strides=1, padding='same', name=None):
    

    conv1 = Conv1D(filters, kernel_size, strides=strides, padding=padding,
                  use_bias=True, kernel_initializer='glorot_normal', name=name)(ip)
    
    reshape = Reshape((t_n, 1, filters))(conv1)
    
    return reshape



def _conv_layer1r(ip, t_n, f_n, filters, kernel_size, strides=1, padding='same', name=None):
    
    reshape1 = Reshape((t_n, f_n))(ip)
    

    conv1 = Conv1D(filters, kernel_size, strides=strides, padding=padding,
                  use_bias=True, kernel_initializer='glorot_normal', name=name)(reshape1)
    
    reshape2 = Reshape((t_n, 1, filters))(conv1)
    
    return reshape2


def _normalize_depth_vars(depth_k, depth_v, filters):

    if type(depth_k) == float:
        depth_k = int(filters * depth_k)
    else:
        depth_k = int(depth_k)

    if type(depth_v) == float:
        depth_v = int(filters * depth_v)
    else:
        depth_v = int(depth_v)

    return depth_k, depth_v