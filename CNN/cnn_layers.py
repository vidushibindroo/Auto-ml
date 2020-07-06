import numpy as np
import math
import sys

eps = 1e-8

"""Defining layers for a fully connected conv 3×3 neural net"""

class fc:
##Defining a class fully connected layer##
    """A fully connected layer takes the output of convolution/pooling and predicts the 
    best label to describe the image"""

    def fc_forward(self, X, W, b):
        out = X @ W + b
        cache = (W, X)
        return out, cache
    
    def fc_backward(self, dout, cache):
        W, h = cache
        dW = h.T @ dout
        db = np.sum(dout, axis=0)
        dX = dout @ W.T
        return dX, dW, db


##Defining various activation functions##
class relu:
##defining activation function relu (Rectified Linear Units)##
    def forward(self, X):
        out = np.maximum(X, 0)
        cache = X
        return out, cache
    
    def backward(self, dout, cache):
        dX = dout.copy()
        dX[cache <= 0] = 0
        return dX


class lrelu:
##defining activation function leaky relu, can be used in case to solve the problem of dying relu##    
    def forward(X, a=1e-3):
        out = np.maximum(a * X, X)
        cache = (X, a)
        return out, cache
    
    def backward(dout, cache):
        X, a = cache
        dX = dout.copy()
        dX[X < 0] *= a
        return dX


def sigmoid_func(X):
    return 1. / (1 + np.exp(-X))


class sigmoid:
##defining activation function sigmoid##
    def forward(self, X):
        out = sigmoid_func(X)
        cache = out
        return out, cache
    
    def backward(self, dout, cache):
        return cache * (1. - cache) * dout


class tanh:
##defining activation function tanh##
    def forward(self, X):
        out = np.tanh(X)
        cache = out
        return out, cache
    
    def backward(self, dout, cache):
        dX = (1 - cache**2) * dout
        return dX


class dropout:
##defining dropout layer##
    def forward(X, p_dropout):
        u = np.random.binomial(1, p_dropout, size=X.shape) / p_dropout
        out = X * u
        cache = u
        return out, cache
    
    def backward(dout, cache):
        dX = dout * cache
        return dX


def ema(running, new, gamma=.9):
##ema is exponential moving average##
    return gamma * running + (1. - gamma) * new



class batch_norm:
##defining batch normalization##
    def forward(self, X, gamma, beta, cache, momentum=.9, train=True):
        running_mean, running_var = cache
        
        if train:
            mu = np.mean(X, axis=0)
            var = np.var(X, axis=0)
            
            X_norm = (X - mu) / np.sqrt(var + eps)
            out = gamma * X_norm + beta
            
            cache = (X, X_norm, mu, var, gamma, beta)
            
            running_mean = ema(running_mean, mu, momentum)
            running_var = ema(running_var, var, momentum)
            
        else:
            X_norm = (X - running_mean) / np.sqrt(running_var + eps)
            out = gamma * X_norm + beta
            cache = None
        
        return out, cache, running_mean, running_var
    
    def backward(self, dout, cache):
        
        X, X_norm, mu, var, gamma, beta = cache
        N, D = X.shape
        
        X_mu = X - mu
        std_inv = 1. / np.sqrt(var + eps)
        
        dX_norm = dout * gamma
        dvar = np.sum(dX_norm * X_mu, axis=0) * -.5 * std_inv**3
        dmu = np.sum(dX_norm * -std_inv, axis=0) + dvar * np.mean(-2. * X_mu, axis=0)
        
        dX = (dX_norm * std_inv) + (dvar * 2 * X_mu / N) + (dmu / N)
        dgamma = np.sum(dout * X_norm, axis=0)
        dbeta = np.sum(dout, axis=0)
        
        return dX, dgamma, dbeta


"""We will approach the conv layer as kind of normal feed forward layer, which is just the
matrix multiplication between the input and the weight. To do this, we will use a utility 
function called im2col, which essentially will stretch our input image depending on the 
filter, stride, and width. Instead of naively looping over the image to do convolution operation 
on our filter on the image and then taking the dot product at each location which will be 
equivalent to the filter size, we'll be gathering all the possible locations that we can 
apply our filter at, then do a single matrix multiplication to get the dot product at each
of those possible locations."""


def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
##First figure out what the size of the output should be##
    N, C, H, W = x_shape
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_height) % stride == 0
    out_height = int((H + 2 * padding - field_height) / stride + 1)
    out_width = int((W + 2 * padding - field_width) / stride + 1)

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return (k.astype(int), i.astype(int), j.astype(int))


def im2col_indices(x, field_height, field_width, padding=1, stride=1):
##Zero-pad the input##
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols


"""After using im2col in conv layer, we’re getting the gradient of the stretched image. 
To undo this, and getting the real image gradient, we’re going to de-im2col that. We’re going to 
apply the operation defined as col2im to the stretched image. And now we have our image 
input gradient"""

def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1,
                   stride=1):
   
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding, stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]



"""The conv layer will accept an input in X: DxCxHxW dimension, input filter 
    W: NFxCxHFxHW, and bias b: Fx1, where:

    D is the number of input
    C is the number of image channel
    H is the height of image
    W is the width of the image
    NF is the number of filter in the filter map W
    HF is the height of the filter, and finally
    HW is the width of the filter.
"""
class conv:
##defining class for a convolution layer##
    def forward(self, X, W, b, stride=1, padding=1):
        cache = W, b, stride, padding
        n_filters, d_filter, h_filter, w_filter = W.shape
        n_x, d_x, h_x, w_x = X.shape
        h_out = (h_x - h_filter + 2 * padding) / stride + 1
        w_out = (w_x - w_filter + 2 * padding) / stride + 1
        
        if not h_out.is_integer() or not w_out.is_integer():
            raise Exception('Invalid output dimension!')
        
        h_out, w_out = int(h_out), int(w_out)
        X_col = im2col_indices(X, h_filter, w_filter, padding=padding, stride=stride)
        W_col = W.reshape(n_filters, -1)
        
        out = W_col @ X_col + b
        out = out.reshape(n_filters, h_out, w_out, n_x)
        out = out.transpose(3, 0, 1, 2)
        
        cache = (X, W, b, stride, padding, X_col)
        
        return out, cache
    
    def backward(self, dout, cache):
        X, W, b, stride, padding, X_col = cache
        n_filter, d_filter, h_filter, w_filter = W.shape
        ##defining bias gradient##
        db = np.sum(dout, axis=(0, 2, 3))
        db = db.reshape(n_filter, -1)

        dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(n_filter, -1)
        dW = dout_reshaped @ X_col.T
        dW = dW.reshape(W.shape)
        
        W_reshape = W.reshape(n_filter, -1)
        dX_col = W_reshape.T @ dout_reshaped
        dX = col2im_indices(dX_col, X.shape, h_filter, w_filter, padding=padding, stride=stride)
        
        return dX, dW, db


##defining maxpool and average pool pooling layers##
def maxpool_forward(X, size=2, stride=2):
    def maxpool(X_col):
        max_idx = np.argmax(X_col, axis=0)
        out = X_col[max_idx, range(max_idx.size)]
        return out, max_idx

    return _pool_forward(X, maxpool, size, stride)


def avgpool_forward(X, size=2, stride=2):
    def avgpool(X_col):
        out = np.mean(X_col, axis=0)
        cache = None
        return out, cache

    return _pool_forward(X, avgpool, size, stride)


def avgpool_backward(dout, cache):
    def davgpool(dX_col, dout_col, pool_cache):
        dX_col[:, range(dout_col.size)] = 1. / dX_col.shape[0] * dout_col
        return dX_col

    return _pool_backward(dout, davgpool, cache)



def _pool_forward(X, pool_fun, size=2, stride=2):
    n, d, h, w = X.shape
    h_out = (h - size) / stride + 1
    w_out = (w - size) / stride + 1

    if not w_out.is_integer() or not h_out.is_integer():
        raise Exception('Invalid output dimension!')

    h_out, w_out = int(h_out), int(w_out)

    X_reshaped = X.reshape(n * d, 1, h, w)
    X_col = im2col_indices(X_reshaped, size, size, padding=0, stride=stride)

    out, pool_cache = pool_fun(X_col)

    out = out.reshape(h_out, w_out, n, d)
    out = out.transpose(2, 3, 0, 1)

    cache = (X, size, stride, X_col, pool_cache)

    return out, cache


def _pool_backward(dout, dpool_fun, cache):
    X, size, stride, X_col, pool_cache = cache
    n, d, w, h = X.shape

    dX_col = np.zeros_like(X_col)
    dout_col = dout.transpose(2, 3, 0, 1).ravel()

    dX = dpool_fun(dX_col, dout_col, pool_cache)

    dX = col2im_indices(dX_col, (n * d, 1, h, w), size, size, padding=0, stride=stride)
    dX = dX.reshape(X.shape)

    return dX
