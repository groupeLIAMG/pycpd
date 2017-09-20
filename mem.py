#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''

@Book{wu97,
  title     = {The Maximum Entropy Method},
  publisher = {Springer},
  year      = {1997},
  author    = {Nailong Wu},
  editor    = {Thomas S. Huang and Teuvo Kohonen and Manfred R. Schroeder},
  series    = {Springer series in information sciences},
  isbn      = {3-540-61965-8},
}
'''

import numpy as np
import pyfftw.interfaces.numpy_fft as np_fft

def lim_malik(data, nitermax=50):
    
    epsilon = 0.01
    gamma = 0.5
    gamma_p = 0.5
    alpha = 0.0
    beta = 0.0
    
    
    
    m, n = data.shape
    
    # compute autocorrelation function
    R = np.abs(np_fft.ifft2(np_fft.fft2(data) * np.conj(np_fft.fft2(data)))) / (n*m)
    
    # we do not (yet) fftshift the data
    
    # initialize arrays
    Cm = np.zeros(R.shape)
    Rx = np.zeros(R.shape)
    Sm = np.zeros(R.shape)
    
    Cm[0, 0] = 1.0 / R[0, 0]
    Sm[:,:] = R[0, 0]
    Rx[0, 0] = R[0, 0]
    
    den = np.sum( R**2 )
    diff = np.sqrt( np.sum( (Rx-R)**2 ) / den )
    diff_pre = diff
    
    n = 0
    while diff > epsilon and n < nitermax:
        #print(n, diff)
        
        if diff > diff_pre:
            # decrease gamma, gamma_prime
            gamma *= 0.5
            gamma_p *= 0.5
            
            
        # eq 2.4.24
        tmp = np_fft.fft2( R - Rx )
        ind = tmp < 0
        if np.sum(ind) > 0:
            tmp = np_fft.fft2(Rx) / np.abs( tmp )
            tmp = 1.0 - gamma * np.min(tmp[ind])
            alpha = max([alpha, tmp])
            
        # eq 2.4.20
        Ry = Rx + (1. - alpha) * (R - Rx)
        
        
        C_p = np_fft.ifft2( 1. / np_fft.fft2( Ry ) )
        
        #eq 2.4.26
        tmp = np_fft.fft2( C_p )
        ind = tmp < 0
        if np.sum(ind) > 0:
            
            tmp = np.abs( tmp )
            tmp = tmp / (np_fft.fft2(Cm) + tmp )
            beta_min = np.max( tmp[ind] )
            beta = max([beta, beta_min + gamma_p*(1.0 - beta_min)])
        
        Cm = beta * Cm + (1.-beta)*C_p
        
        Rx = np_fft.ifft2( 1. / np_fft.fft2(Cm) )
        
        diff_pre = diff
        diff = np.sqrt( np.sum( (Rx-R)**2 ) / den )
        
        n += 1
        
    S = np_fft.fftshift( np_fft.fft2( Rx ) )
    return S
        
        
if __name__ == '__main__':
    from skimage.io import imread
    from skimage import data_dir
    from skimage.transform import rescale

    import matplotlib.pyplot as plt

    image = imread(data_dir + "/phantom.png", as_grey=True)
    image = rescale(image, scale=0.4, mode='reflect')
    
    S = lim_malik(image)
    
    plt.imshow(np.log10(np.abs(S)))
    plt.colorbar()
    plt.show()
    
    TF2D = np.abs(np_fft.fft2(image))
    TF2D = np_fft.fftshift(TF2D)
    
    plt.imshow(np.log10(np.abs(TF2D)))
    plt.colorbar()
    plt.show()
    