# -*- coding: utf-8 -*-
"""

Copyright 2017 Bernard Giroux

This file is part of pycpd.

BhTomoPy is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it /will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
cimport numpy as np

from scipy.sparse import csr_matrix

cimport cradon

def radon2d(data, theta):
    
    if np.min(theta) < 0.0 or np.max(theta) >= np.pi:
        raise ValueError('theta should be within [0 pi)')
    
    nx, ny = data.shape
    if nx != ny:
        raise RuntimeError('data should be a square array')
    nw = nx
    
    sinogram = np.zeros((nw,theta.size))
    
    Tx = np.zeros((nw,2))
    Rx = np.zeros((nw,2))
    
    cdef size_t nTx = Tx.shape[0]

    
    for nt in range(theta.size):
            
        if theta[nt] == 0.0:
            for n in range(nw):
                Tx[n,0] = n
                Tx[n,1] = 0
                Rx[n,0] = n
                Rx[n,1] = ny-1
    
        elif np.abs(theta[nt] - np.pi/2) < 1.e-10:
    
            for n in range(nw):
                Tx[n,0] = 0
                Tx[n,1] = n
                Rx[n,0] = nx-1
                Rx[n,1] = n

        elif theta[nt] < np.pi/2:
            
            xs = nw/2 * (1. - np.cos(theta[nt]) )
            ys = nw/2 * (1. - np.sin(theta[nt]) )
            
            for n in range(nw):
                xk = xs + n * np.cos(theta[nt])
                yk = ys + n * np.sin(theta[nt])
    
                x0 = 0.0
                y0 = yk + (xk-x0) / np.tan(theta[nt])
                if y0 > ny-1:
                    y0 = ny-1
                    x0 = xk - (y0-yk) * np.tan(theta[nt])
                x1 = nx-1
                y1 = yk - (x1-xk) / np.tan(theta[nt])
                if y1 < 0.0:
                    y1 = 0.0
                    x1 = xk + (yk-y1) * np.tan(theta[nt])
    
                Tx[n,0] = x0
                Tx[n,1] = y0
                Rx[n,0] = x1
                Rx[n,1] = y1
    
    
        else:
    
            xs = nw/2 * (1. - np.cos(theta[nt]) )
            ys = nw/2 * (1. - np.sin(theta[nt]) )
            
            for n in range(nw):
                xk = xs + n * np.cos(theta[nt])
                yk = ys + n * np.sin(theta[nt])
    
                x0 = nx-1
                y0 = yk + (x0-xk) * np.tan(theta[nt]-np.pi/2.0)
                if y0 > ny-1:
                    y0 = ny-1
                    x0 = xk + (y0-yk) / np.tan(theta[nt]-np.pi/2.0)
                x1 = 0.0
                y1 = yk - (xk-x1) * np.tan(theta[nt]-np.pi/2.0)
                if y1 < 0.0:
                    y1 = 0.0
                    x1 = xk - (yk-y1) / np.tan(theta[nt]-np.pi/2.0)
    
                Tx[n,0] = x0
                Tx[n,1] = y0
                Rx[n,0] = x1
                Rx[n,1] = y1
    
        if np.max(Rx) > nx-1:
            print('Rx '+str(n))
            print(Rx)
        if np.max(Tx) > nx-1:
            print('Tx '+str(n))
            print(Tx)
    
    
        Ldata = ([0.0], [0.0], [0.0])
    
        cradon.radon2d(<double*> np.PyArray_DATA(Tx), <double*> np.PyArray_DATA(Rx), nTx, nx, ny, Ldata)
    
        M = nTx
        N = nx * ny
        L = csr_matrix(Ldata, shape=(M,N))

        p = L.dot(data.flatten())
        sinogram[:,nt] = p




    return sinogram
