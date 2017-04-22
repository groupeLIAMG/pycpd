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
'''
Classes and functions to compute the Curie-point depth

B. Giroux
INRS-ETE
'''
import struct
import numpy as np
from scipy.special import gamma, kv, lambertw
from scipy.optimize import fmin, fminbound, fmin_cobyla
from scipy.signal import tukey
import pyproj
import netCDF4

import time
import warnings

import geostat


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Polygon:
    def __init__(self):
        self.pts = []

    def addPoint(self, p):
        self.pts.append(p)

    def inside(self, p):
        test = False

        j = len(self.pts) - 1
        for i in range(j+1):
            if ( (self.pts[i].y >= p.y ) != (self.pts[j].y >= p.y) ) and \
            (p.x <= (self.pts[j].x - self.pts[i].x) * \
                (p.y - self.pts[i].y) / (self.pts[j].y - self.pts[i].y) + self.pts[i].x):
                test = not test
            j = i
        return test



class Forage:
    def __init__(self, lat=0.0, lon=0.0, proj4string=None):
        self.lat = lat
        self.lon = lon
        self.x = []
        self.y = []
        self.dom_id = 0
        self.site_name = ''
        self.Q0 = []                 # mW/m2
        self.k = []                  # W/m/K
        self.k_sim = []              # W/m/K
        self.A = []                  # W/m3 x 1e-6
        self.A_sim = []              # W/m3 x 1e-6
        self.zb_sim = []             # m
        self.S_r = []                # Radial spectrum
        self.k_r =[]                 # wavenumber of Spectrum
        self.E2 = []
        self.pad = False
        self.beta_sim = []
        self.zt_sim = []
        self.C = []
        self.offshore = False
        self.zb_sat = None
        self.updateProj(proj4string)

    def __eq__(self, rhs):  # to search in lists of Forage
        if rhs == None:
            return False
        else:
            return self.is_close(rhs.lat, rhs.lon)

    def is_close(self, lat, lon, tol=0.00001):
        return abs(self.lat-lat)<tol and abs(self.lon-lon)<tol

    def updateProj(self, proj4string):
        if proj4string != None:
            lcc = pyproj.Proj(proj4string)
            self.x, self.y = lcc(self.lon, self.lat)
        else:
            warnings.warn('Projection undefined, coordinates not projected', UserWarning)

def proj4string2dict(proj4string):
    '''
    Function to build the set of arguments to the constructor of a matplotlib Basemap object
    from a proj4 string

    This is far from complete and was tested only with the Lambert Conformal projection
    '''
    params = {}
    tmp = proj4string.split('+')
    a=None
    b=None
    for t in tmp:
        if len(t)>0:
            t2 = t.strip().split('=')
            if t2[0] == 'proj':
                params['projection'] = t2[1]
            elif t2[0]=='lon_0' or t2[0]=='lon_1' or t2[0]=='lon_2' or t2[0]=='lat_0' \
              or t2[0]=='lat_1' or t2[0]=='lat_2' or t2[0]=='lat_ts' or t2[0]=='k_0':
                params[t2[0]] = float(t2[1])
            elif t2[0]=='ellps':
                params[t2[0]] = t2[1]
            elif t2[0]=='a':
                a = float(t2[1])
            elif t2[0]=='b':
                b = float(t2[1])

    if a != None and b != None:
        params['rsphere'] = (a,b)
    return params

class Grid2d:
    def __init__(self, proj4string=''):
        self.data = []
        self.ncol = 0
        self.nrow = 0
        self.xwest = 0
        self.xeast = 0
        self.ysouth = 0
        self.ynorth = 0
        self.dx = 0
        self.dy = 0
        self.G = None
        self.latlon = False
#        self.proj4string = '+proj=lcc +lat_1=60 +lat_2=46 +lat_0=44 +lon_0=-68.5 +x_0=0 +y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs'  # EPSG:32198
        self.proj4string = proj4string


    def inside(self, pt, buffer=0.0):
        return pt.x>=(self.xwest+buffer) and pt.x<=(self.xeast-buffer) and pt.y>=(self.ysouth+buffer) and pt.y<=(self.ynorth-buffer)

    def as_2col(self):
        x = self.xwest + self.dx * np.arange(self.ncol)
        y = self.ysouth + self.dy * np.arange(self.nrow)
        x = x.reshape(-1,1)
        y = y.reshape(-1,1)
        return np.hstack((np.kron(np.ones((self.nrow,1)), x),
                          np.kron(y, np.ones((self.ncol,1)))))


    def readUSGSfile(self, fname):
        """
        Read grid in USGS format
        """
        with open(fname, 'rb') as ifile:
            ifile.seek(68)
            self.ncol = struct.unpack('>i', ifile.read(4))[0]
            self.nrow = struct.unpack('>i', ifile.read(4))[0]
            ifile.seek(4, 1)
            self.xwest = struct.unpack('>f', ifile.read(4))[0]
            self.dx = struct.unpack('>f', ifile.read(4))[0]
            self.xeast = self.xwest + self.dx*(self.ncol-1)
            self.ysouth = struct.unpack('>f', ifile.read(4))[0]
            self.dy = struct.unpack('>f', ifile.read(4))[0]
            self.ynorth = self.ysouth + self.dy*(self.nrow-1)
            ifile.seek(24, 1)
            pos = ifile.tell()
            ifile.seek(0,2)
            pos2 = ifile.tell()
            nd = int( (pos2-pos)/4 )
            ifile.seek(pos, 0)
            A = np.array(struct.unpack('>'+str(nd)+'f', ifile.read(nd*4)))
            A[A>1.e+31]=np.nan
            A[A<-1.e+31]=np.nan
            nmiss = (self.ncol+3)*self.nrow - A.size
            if nmiss>0:
                A = np.append(A, np.zeros((nmiss,)))
            A = A.reshape(self.nrow, self.ncol+3)
            self.data = A[:,:-3]

            try :
                lcc = pyproj.Proj(self.proj4string)
                self.c0 = lcc(self.xwest, self.ysouth, inverse=True)
                self.c1 = lcc(self.xeast, self.ynorth, inverse=True)
            except RuntimeError as e:
                raise e

    def ncdump(self, fname='test.nc'):
        dataset = netCDF4.Dataset(fname, 'w', format='NETCDF4')

        dataset.createDimension('x', self.ncol)
        dataset.createDimension('y', self.nrow)

        varx = dataset.createVariable('x', np.float64, ('x'))
        vary = dataset.createVariable('y', np.float64, ('y'))
        varz = dataset.createVariable('z', np.float32, ('y','x'),
                                      zlib=True, complevel=3, fill_value = np.nan)
        varz.actual_range = [np.min(self.data), np.max(self.data)]

        varx[:] = self.xwest + self.dx*np.arange(self.ncol, dtype=np.float64)
        vary[:] = self.ysouth + self.dy*np.arange(self.nrow, dtype=np.float64)
        varz[:] = np.array(self.data, dtype=np.float32)

        dataset.close()

    def readnc(self, fname):
        try:
            dataset = netCDF4.Dataset(fname, 'r', format='NETCDF4')
        except OSError:
            raise IOError('Could not open '+fname)

        if 'x_range' in dataset.variables:
            x = dataset.variables['x_range']
            self.xwest = x[0]
            self.xeast = x[1]
            y = dataset.variables['y_range']
            self.ysouth = y[0]
            self.ynorth = y[1]
            nd = dataset.variables['dimension']
            self.ncol = nd[0]
            self.nrow = nd[1]
            dx = dataset.variables['spacing']
            self.dx = dx[0]
            self.dy = dx[1]
            self.data = np.array(dataset.variables['z']).reshape((self.nrow,self.ncol))
            self.data = self.data[::-1,:]
        else:
            try:
                x = dataset.variables['x']
            except KeyError:
                x = dataset.variables['lon']
                self.latlon = True
            self.xwest = x[0]
            self.xeast = x[-1]
            self.ncol = x.size
            self.dx = x[1] - x[0]
            try:
                y = dataset.variables['y']
            except KeyError:
                y = dataset.variables['lat']
            self.ysouth = y[0]
            self.ynorth = y[-1]
            self.nrow = y.size
            self.dy = y[1] - y[0]
            self.data = np.array(dataset.variables['z'])
        dataset.close()

        try :
            lcc = pyproj.Proj(self.proj4string)
            self.c0 = lcc(self.xwest, self.ysouth, inverse=True)
            self.c1 = lcc(self.xeast, self.ynorth, inverse=True)
        except RuntimeError as e:
            raise e

    def preFFTMA(self, cm):
        """
        Compute matrix G for FFT-MA simulations

        INPUT
            cm: list of covariance models

        """
        small = 1.0e-6
        Nx = 2*self.ncol
        Ny = 2*self.nrow

        Nx2 = Nx/2
        Ny2 = Ny/2

        x = self.dx * np.hstack((np.arange(Nx2), np.arange(-Nx2+1,1)))
        y = self.dy * np.hstack((np.arange(Ny2), np.arange(-Ny2+1,1)))

        x = np.kron(x,np.ones((Ny,)))
        y = np.kron(y,np.ones((1,Nx)).T).flatten()

        d = 0
        for c in cm:
            d = d + c.compute(np.vstack((x,y)).T, np.zeros((1,2)))
        K = d.reshape(Nx,Ny)

        mk = True
        while mk:
            mk = False
            if np.min(K[0,:])>small:
                # Enlarge grid to make sure that covariance falls to zero
                Ny = 2*Ny
                mk = True

            if np.min(K[:,0])>small:
                Nx = 2*Nx
                mk = True

            if mk:
                Nx2 = Nx/2
                Ny2 = Ny/2

                x = self.dx * np.hstack((np.arange(Nx2), np.arange(-Nx2+1,1)))
                y = self.dy * np.hstack((np.arange(Ny2), np.arange(-Ny2+1,1)))

                x = np.kron(x,np.ones((Ny,)))
                y = np.kron(y,np.ones((1,Nx)).T).flatten()

                d = 0
                for c in cm:
                    d = d + c.compute(np.vstack((x,y)).T, np.zeros((1,2)))
                K = d.reshape(Nx,Ny)

        self.G = np.sqrt(np.fft.fft2(K))

    def FFTMA(self):
        """
        Perform FFT-MA simulation using pre-computed spectral matrix

        OUTPUT
            Z: simulated field of size nx x ny
        """
        if self.G is None:
            raise RuntimeError('Spectral matrix G should be precomputed')

        Nx,Ny = self.G.shape
        U = np.fft.fft2(np.random.randn(self.G.shape[0], self.G.shape[1]))

        Z = np.real(np.fft.ifft2(self.G*U))
        return Z[:self.ncol, :self.nrow].T  # transpose to have nx cols & ny rows


    def getRadialSpectrum(self, xc, yc, ww, window=np.hanning, detrend=0, scalFac=0.001):
        """
        Compute radial spectrum for point at (xc,yc) for square window of
        width equal to ww

        Parameters
        ----------
        xc       : X coordinate of point
        yc       : Y coordinate of point
        ww       : window width
        window   : padding window
        detrend  : control detrending of data
                   0 : data unchanged (default)
                   1 : remove best-fitting linear trend
                   2 : remove mean value
                   3 : remove median value
                   4 : remove mid value, i.e. 0.5 * (max + min)
        scalFac  : scaling factor to get k in rad/km  (0.001 by default, for grid in m)

        Returns
        -------
        S       : Radial spectrum
        k       : wavenumber [rad/km]
        E2      : variance of S
        flagPad : True if zero padding was applied
        """

        # check if in grid
        if xc<self.xwest or xc>self.xeast or yc<self.ysouth or yc>self.ynorth:
            raise ValueError('Point outside grid')

        if self.dx != self.dy:
            raise RuntimeError('Grid cell size should be equal in x and y')

        # get subgrid
        nw = 1+int(ww/self.dx)
        nw2 = int(nw/2)
        ix = int((xc-self.xwest)/self.dx)
        iy = int((yc-self.ysouth)/self.dx)

        imin = ix-nw2
        imax = ix+nw2+1
        jmin = iy-nw2
        jmax = iy+nw2+1
        flagPad = False
        if imin < 0:
            print('Warning: pt close to eastern edge, padding will be applied ('+str(-imin)+' cols)')
            imin = 0
            flagPad = True
        if imax > self.ncol:
            print('Warning: pt close to western edge, padding will be applied ('+str(imax-self.ncol)+' cols)')
            imax = self.ncol
            flagPad = True
        if jmin < 0:
            print('Warning: pt close to southern edge, padding will be applied ('+str(-jmin)+' rows)')
            jmin = 0
            flagPad = True
        if jmax > self.nrow:
            print('Warning: pt close to northern edge, padding will be applied ('+str(jmax-self.nrow)+' rows)')
            jmax = self.nrow
            flagPad = True

        data = self.data[jmin:jmax, imin:imax].copy()
        ny,nx = data.shape
        if ny<nw:
            data = np.vstack((data, np.zeros((nw-ny,nx))))  # pad with zeros
        if nx < nw:
            data = np.hstack((data, np.zeros((nw,nw-nx))))

        if detrend == 1:
            # remove linear trend
            x,y = np.meshgrid(np.arange(data.shape[1]),np.arange(data.shape[0]))

            A = np.column_stack((x.flatten(), y.flatten(), np.ones(x.size)))
            c, resid, rank, sigma = np.linalg.lstsq(A,data.flatten())

            p = np.dot(A, c)
            for n in range(data.size):
                data.flat[n] -= p[n]

        elif detrend == 2:
            data = data - np.mean(data)
        elif detrend == 3:
            data = data - np.median(data)
        elif detrend == 4:
            mid = 0.5 * (data.max() - data.min())
            data = data - mid

        taper = np.ones(data.shape)

        if window is tukey:
            ht = window(taper.shape[0], alpha=0.05)  # Bouligand uses 5% tapering
        else:
            ht = window(taper.shape[0])
        for n in range(taper.shape[1]):
            taper[:,n] *= ht
        if window is tukey:
            ht = window(taper.shape[1], alpha=0.05)
        else:
            ht = window(taper.shape[1])
        for n in range(taper.shape[0]):
            taper[n,:] *= ht

        TF2D = np.abs(np.fft.fft2(data*taper))
        TF2D = np.fft.fftshift(TF2D)

        dx = self.dx * scalFac  # from m to km

        dk = 2.0*np.pi / (nw-1) / dx
        kbins = np.arange(dk, dk*nw/2, dk)
        nbins = kbins.size-1
        S = np.zeros((nbins,))
        k = np.zeros((nbins,))
        E2 = np.zeros((nbins,))

        i0 = int((nw-1)/2)
        iy,ix= np.meshgrid(np.arange(nw), np.arange(nw))
        kk = np.sqrt( ((ix-i0)*dk)**2 + ((iy-i0)*dk)**2 )

        for n in range(nbins):
            ind = np.logical_and( kk>=kbins[n], kk<=kbins[n+1] )
            rr = 2.0*np.log(TF2D[ind])
            S[n] = np.mean(rr)
            k[n] = np.mean(kk[ind])
            E2[n] = np.std(rr) / np.sqrt( rr.size/2 )

        return S, k, E2, flagPad


def bouligand4(beta, zt, dz, kh, C=0.0):
    '''
    Equation (4) of Bouligand et al. (2009)

    Parameters
    ----------
    beta : fractal parameter
    zt   : top of magnetic sources
    dz   : thickness of magnetic sources
    kh   : norm of the wave number in the horizontal plane
    C    : field constant (Maus et al., 1997)

    Returns
    -------
    Radial power spectrum of magnetic anomalies

    Reference papers
    ---------------
    Bouligand, C., J. M. G. Glen, and R. J. Blakely (2009), Mapping Curie
      temperature depth in the western United States with a fractal model for
      crustal magnetization, J. Geophys. Res., 114, B11104,
      doi:10.1029/2009JB006494
    Maus, S., D. Gordon, and D. Fairhead (1997), Curie temperature depth
      estimation using a self-similar magnetization model, Geophys. J. Int.,
      129, 163–168, doi:10.1111/j.1365-246X.1997.tb00945.x
    '''
    khdz = kh*dz
    coshkhdz = np.cosh(khdz)

    Phi1d = C - 2.0*kh*zt - (beta-1.0)*np.log(kh) - khdz
    A = np.sqrt(np.pi)/gamma(1.0+0.5*beta) * (0.5*coshkhdz*gamma(0.5*(1.0+beta))
                                              - kv((-0.5*(1.0+beta)), khdz) * np.power(0.5*khdz,(0.5*(1.0+beta)) ))
    Phi1d += np.log(A)
    return Phi1d

def find_beta(zb, Phi_exp, kh, beta0, zt=1.0, C=0, wlf=False):
    '''
    Find fractal parameter beta for a given radial spectrum

    Parameters
    ----------
        Phi_exp : Spectrum values for kh
        kh      : norm of the wave number in the horizontal plane
        beta0   : starting value
        zt      : depth of top of magnetic layer
        C       : field constant (Maus et al., 1997)
        wlf     : apply low frequency weighting

    Returns
    -------
        beta, misfit
    '''
    if wlf:
        w = np.linspace(2.0, 1.0, Phi_exp.size)
        w /= w.sum()
    else:
        w = 1.0
    # define function to minimize
    def func(beta, zb, Phi_exp, zt, kh, C):
        dz = zb - zt
        return np.linalg.norm(w*(Phi_exp - bouligand4(beta, zt, dz, kh, C)))

    beta_opt = fmin(func, x0=beta0, args=(zb, Phi_exp, zt, kh, C), full_output=True, disp=False)
    return beta_opt[0][0],  beta_opt[1]

def find_zt(zb, Phi_exp, kh, beta, zt0, C=0, wlf=False):
    '''
    Find depth of top of magnetic layer for a given radial spectrum

    Parameters
    ----------
        Phi_exp : Spectrum values for kh
        kh      : norm of the wave number in the horizontal plane
        beta    : fractal paremeter
        zt0     : starting value
        C       : field constant (Maus et al., 1997)
        wlf     : apply low frequency weighting

    Returns
    -------
       zt, misfit
    '''
    if wlf:
        w = np.linspace(2.0, 1.0, Phi_exp.size)
        w /= w.sum()
    else:
        w = 1.0
    # define function to minimize
    def func(zt, zb, Phi_exp, beta, kh, C):
        dz = zb - zt
        return np.linalg.norm(w*(Phi_exp - bouligand4(beta, zt, dz, kh, C)))

    xopt = fmin(func, x0=zt0, args=(zb, Phi_exp, beta, kh, C), full_output=True, disp=False)
    return xopt[0][0], xopt[1]

def find_dz(dz0, Phi_exp, kh, beta, zt, C, wlf=False):
    '''
    Find thichness of magnetic layer for a given radial spectrum

    Parameters
    ----------
        dz0     : starting value
        Phi_exp : Spectrum values for kh
        kh      : norm of the wave number in the horizontal plane
        beta    : fractal paremeter
        zt      : depth of top of magnetic layer
        C       : field constant (Maus et al., 1997)
        wlf     : apply low frequency weighting

    Returns
    -------
       dz, misfit
    '''
    if wlf:
        w = np.linspace(2.0, 1.0, Phi_exp.size)
        w /= w.sum()
    else:
        w = 1.0
    # define function to minimize
    def func(dz, zt, Phi_exp, beta, kh, C):
        return np.linalg.norm(w*(Phi_exp - bouligand4(beta, zt, dz, kh, C)))

    xopt = fmin(func, x0=dz0, args=(zt, Phi_exp, beta, kh, C), full_output=True, disp=False)
    return xopt[0][0], xopt[1]

def find_C(dz, Phi_exp, kh, beta, zt, C0, wlf=False):
    '''
    Find field constant for a given radial spectrum

    Parameters
    ----------
        dz      : thichness of magnetic layer
        Phi_exp : Spectrum values for kh
        kh      : norm of the wave number in the horizontal plane
        beta    : fractal paremeter
        zt      : depth of top of magnetic layer
        C0      : starting value of field constant (Maus et al., 1997)
        wlf     : apply low frequency weighting

    Returns
    -------
       C, misfit
    '''
    if wlf:
        w = np.linspace(2.0, 1.0, Phi_exp.size)
        w /= w.sum()
    else:
        w = 1.0
    # define function to minimize
    def func(C, zt, Phi_exp, beta, kh, dz):
        return np.linalg.norm(w*(Phi_exp - bouligand4(beta, zt, dz, kh, C)))

    xopt = fmin(func, x0=C0, args=(zt, Phi_exp, beta, kh, dz), full_output=True, disp=False)
    return xopt[0][0], xopt[1]


def find_beta_zt_dz_C(Phi_exp, kh, beta0, zt0, dz0, C0, wlf=False):
    '''
    Find fractal model parameters for a given radial spectrum

    Parameters
    ----------
        Phi_exp : Spectrum values for kh
        kh      : norm of the wave number in the horizontal plane
        beta0   : starting value of beta
        zt0     : starting value of depth of top of magnetic layer
        dz0     : starting value of layer thickness
        C0      : starting value of field constant (Maus et al., 1997)
        wlf     : apply low frequency weighting

    Returns
    -------
        beta, zt, dz, C, misfit
    '''
    if wlf:
        w = np.linspace(2.0, 1.0, Phi_exp.size)
        w /= w.sum()
    else:
        w = 1.0
    # define function to minimize
    def func(x, Phi_exp, kh):
        beta = x[0]
        zt = x[1]
        dz = x[2]
        C = x[3]
        return np.linalg.norm(w*(Phi_exp - bouligand4(beta, zt, dz, kh, C)))

    xopt = fmin(func, x0=np.array([beta0, zt0, dz0, C0]), args=(Phi_exp, kh), full_output=True, disp=False)
    beta_opt = xopt[0][0]
    zt_opt = xopt[0][1]
    dz_opt = xopt[0][2]
    C_opt = xopt[0][3]
    misfit = xopt[1]
    return beta_opt, zt_opt, dz_opt, C_opt, misfit

def find_beta_zt_C(Phi_exp, kh, beta0, zt0, C0, zb, wlf=False):
    '''
    Find fractal model parameters, depth of top of magnetic layer and
    constant C for a given radial spectrum and depth to bottom value

    Parameters
    ----------
        Phi_exp : Spectrum values for kh
        kh      : norm of the wave number in the horizontal plane
        beta0   : starting value of beta
        zt0     : starting value of depth to top of magnetic layer
        C0      : starting value of field constant (Maus et al., 1997)
        zb      : depth of bottom of magnetic layer
        wlf     : apply low frequency weighting

    Returns
    -------
        beta, zt, C, misfit
    '''
    if wlf:
        w = np.linspace(2.0, 1.0, Phi_exp.size)
        w /= w.sum()
    else:
        w = 1.0
    # define function to minimize
    def func(x, Phi_exp, kh, zb):
        beta = x[0]
        zt = x[1]
        dz = zb-zt
        C = x[2]
        return np.linalg.norm(w*(Phi_exp - bouligand4(beta, zt, dz, kh, C)))

    xopt = fmin(func, x0=np.array([beta0, zt0, C0]), args=(Phi_exp, kh, zb), full_output=True, disp=False)
    beta_opt = xopt[0][0]
    zt_opt = xopt[0][1]
    C_opt = xopt[0][2]
    misfit = xopt[1]
    return beta_opt, zt_opt, C_opt, misfit

def find_beta_zt_C_bound(Phi_exp, kh, beta, zt, C, zb, wlf=False):
    '''
    Find fractal model parameters, depth of top of magnetic layer and
    constant C for a given radial spectrum and depth to bottom value

    Parameters
    ----------
        Phi_exp : Spectrum values for kh
        kh      : norm of the wave number in the horizontal plane
        beta0   : starting value of beta
        zt0     : starting value of depth to top of magnetic layer
        C0      : starting value of field constant (Maus et al., 1997)
        zb      : depth of bottom of magnetic layer
        wlf     : apply low frequency weighting

    Returns
    -------
        beta, zt, C, misfit
    '''
    beta0, beta1, beta2 = beta
    zt0, zt1, zt2 = zt
    C0, C1, C2 = C
    
    if wlf:
        w = np.linspace(2.0, 1.0, Phi_exp.size)
        w /= w.sum()
    else:
        w = 1.0
    # define function to minimize
    def func(x, Phi_exp, kh, zb):
        beta = x[0]
        zt = x[1]
        dz = zb-zt
        C = x[2]
        return np.linalg.norm(w*(Phi_exp - bouligand4(beta, zt, dz, kh, C)))

    def cons1(x):
        return beta2-x[0], zt2-x[1], C2-x[2]
    def cons2(x):
        return x[1]-beta1, x[1]-zt1, x[2]-C1
        
    xopt = fmin_cobyla(func, x0=np.array([beta0, zt0, C0]), cons=(cons1,cons2), args=(Phi_exp, kh, zb), consargs=(), disp=False)
    beta_opt = xopt[0]
    zt_opt = xopt[1]
    C_opt = xopt[2]
    misfit = func(xopt, Phi_exp, kh, zb)
    return beta_opt, zt_opt, C_opt, misfit

def find_beta_zb_C(Phi_exp, kh, beta0, zb0, C0, zt=1.0, wlf=False):
    '''
    Find fractal model parameters, depth of bottom of magnetic layer and
    constant C for a given radial spectrum

    Parameters
    ----------
        Phi_exp : Spectrum values for kh
        kh      : norm of the wave number in the horizontal plane
        beta0   : starting value of beta
        zb0     : starting value of depth to bottom of magnetic layer
        C0      : starting value of field constant (Maus et al., 1997)
        zt      : depth of top of magnetic layer
        wlf     : apply low frequency weighting

    Returns
    -------
        beta, zb, C, misfit
    '''
    if wlf:
        w = np.linspace(2.0, 1.0, Phi_exp.size)
        w /= w.sum()
    else:
        w = 1.0
    # define function to minimize
    def func(x, Phi_exp, kh, zt):
        beta = x[0]
        dz = x[1]-zt
        C = x[2]
        return np.linalg.norm(w*(Phi_exp - bouligand4(beta, zt, dz, kh, C)))

    xopt = fmin(func, x0=np.array([beta0, zb0, C0]), args=(Phi_exp, kh, zt), full_output=True, disp=False)
    beta_opt = xopt[0][0]
    zb_opt = xopt[0][1]
    C_opt = xopt[0][2]
    misfit = xopt[1]
    return beta_opt, zb_opt, C_opt, misfit

def find_beta_zb_zt(Phi_exp, kh, beta0, zb0, zt0, C, wlf=False):
    '''
    Find fractal model parameters, depth of bottom and depth to top of magnetic
    layer for a given radial spectrum

    Parameters
    ----------
        Phi_exp : Spectrum values for kh
        kh      : norm of the wave number in the horizontal plane
        beta0   : starting value of beta
        zb0     : starting value of depth to bottom of magnetic layer
        zt0     : starting value of depth of top of magnetic layer
        C       : field constant (Maus et al., 1997)
        wlf     : apply low frequency weighting

    Returns
    -------
        beta, dz, C, misfit
    '''
    if wlf:
        w = np.linspace(2.0, 1.0, Phi_exp.size)
        w /= w.sum()
    else:
        w = 1.0
    # define function to minimize
    def func(x, Phi_exp, kh, C):
        beta = x[0]
        dz = x[1]-x[2]
        zt = x[2]
        return np.linalg.norm(w*(Phi_exp - bouligand4(beta, zt, dz, kh, C)))

    xopt = fmin(func, x0=np.array([beta0, zb0, zt0]), args=(Phi_exp, kh, C), full_output=True, disp=False)
    beta_opt = xopt[0][0]
    zb_opt = xopt[0][1]
    zt_opt = xopt[0][2]
    misfit = xopt[1]
    return beta_opt, zb_opt, zt_opt, misfit

def find_beta_zt(zb, Phi_exp, kh, beta0, zt0, C=0, wlf=False):
    '''
    Find fractal parameter beta and depth of top of magnetic layer for a given radial spectrum

    Parameters
    ----------
        Phi_exp : Spectrum values for kh
        kh      : norm of the wave number in the horizontal plane
        beta0   : starting value
        zt0     : depth of top of magnetic layer
        C       : field constant (Maus et al., 1997)
        wlf     : apply low frequency weighting

    Returns
    -------
        beta, zt, misfit
    '''
    if wlf:
        w = np.linspace(2.0, 1.0, Phi_exp.size)
        w /= w.sum()
    else:
        w = 1.0
    # define function to minimize
    def func(x, zb, Phi_exp, kh, C):
        beta = x[0]
        zt = x[1]
        dz = zb - zt
        return np.linalg.norm(w*(Phi_exp - bouligand4(beta, zt, dz, kh, C)))

    xopt = fmin(func, x0=np.array([beta0, zt0]), args=(zb, Phi_exp, kh, C), full_output=True, disp=False)
    beta_opt = xopt[0][0]
    zt_opt = xopt[0][1]
    misfit = xopt[1]
    return beta_opt, zt_opt, misfit

def find_beta_C(zb, Phi_exp, kh, beta0, C0, zt=1.0, wlf=False):
    '''
    Find fractal parameter beta and constant C for a given radial spectrum

    Parameters
    ----------
        Phi_exp : Spectrum values for kh
        kh      : norm of the wave number in the horizontal plane
        beta0   : starting value of beta
        C0      : starting value of C
        zt      : depth of top of magnetic layer
        wlf     : apply low frequency weighting

    Returns
    -------
        beta, C, misfit
    '''
    if wlf:
        w = np.linspace(2.0, 1.0, Phi_exp.size)
        w /= w.sum()
    else:
        w = 1.0
    # define function to minimize
    def func(x, zb, Phi_exp, kh, zt):
        beta = x[0]
        C = x[1]
        dz = zb - zt
        return np.linalg.norm(w*(Phi_exp - bouligand4(beta, zt, dz, kh, C)))

    xopt = fmin(func, x0=np.array([beta0, C0]), args=(zb, Phi_exp, kh, zt), full_output=True, disp=False)
    beta_opt = xopt[0][0]
    C_opt = xopt[0][1]
    misfit = xopt[1]
    return beta_opt, C_opt, misfit

def find_zt_dz(Phi_exp, kh, zt0, dz0, beta, C, wlf=False):
    '''
    Find fractal depth of top and thickness of magnetic layer for a given radial spectrum

    Parameters
    ----------
        Phi_exp : Spectrum values for kh
        kh      : norm of the wave number in the horizontal plane
        zt0     : starting value of depth of top of magnetic layer
        dz0     : starting value of thickness
        C       : field constant (Maus et al., 1997)
        wlf     : apply low frequency weighting

    Returns
    -------
        zt, dz, misfit
    '''
    if wlf:
        w = np.linspace(2.0, 1.0, Phi_exp.size)
        w /= w.sum()
    else:
        w = 1.0
    # define function to minimize
    def func(x, Phi_exp, kh, beta, C):
        zt = x[0]
        dz = x[1]
        return np.linalg.norm(w*(Phi_exp - bouligand4(beta, zt, dz, kh, C)))

    xopt = fmin(func, x0=np.array([zt0, dz0]), args=(Phi_exp, kh, beta, C), disp=False, full_output=True)
    zt_opt = xopt[0][0]
    dz_opt = xopt[0][1]
    misfit = xopt[1]
    return zt_opt, dz_opt, misfit

def find_zb(Phi_exp, kh, beta, zt, zb0, C=0.0, wlf=False):
    '''
    Find depth to bottom of magnetic slab for a given radial spectrum

    Parameters
    ----------
        Phi_exp : Spectrum values for kh
        kh      : norm of the wave number in the horizontal plane
        beta    : fractal parameter
        zt      : depth of top of magnetic layer
        zb0     : starting value
        C       : field constant (Maus et al., 1997)
        wlf     : apply low frequency weighting

    Returns
    -------
        zb, misfit
    '''
    if wlf:
        w = np.linspace(2.0, 1.0, Phi_exp.size)
        w /= w.sum()
    else:
        w = 1.0
    # define function to minimize
    def func(zb, beta, Phi_exp, zt, kh, C):
        dz = zb - zt
        return np.linalg.norm(w*(Phi_exp - bouligand4(beta, zt, dz, kh, C)))

    x = fmin(func, x0=zb0, args=(beta, Phi_exp, zt, kh, C), disp=False, full_output=True)
    zb_opt = x[0]
    misfit = x[1]
    return zb_opt[0], misfit

def find_zb_zt_C(Phi_exp, kh, beta, zb0, zt0, C0, wlf=False):
    '''
    Find fractal model parameters for a given radial spectrum

    Parameters
    ----------
        Phi_exp : Spectrum values for kh
        kh      : norm of the wave number in the horizontal plane
        beta    : value of beta
        zb0     : starting value of depth to bottom of magnetic layer
        zt0     : starting value of depth of top of magnetic layer
        C0      : starting value of field constant (Maus et al., 1997)
        wlf     : apply low frequency weighting

    Returns
    -------
        zb, zt, C, misfit
    '''
    if wlf:
        w = np.linspace(2.0, 1.0, Phi_exp.size)
        w /= w.sum()
    else:
        w = 1.0
    # define function to minimize
    def func(x, Phi_exp, kh, beta):
        zb = x[0]
        zt = x[1]
        C = x[2]
        dz = zb - zt
        return np.linalg.norm(w*(Phi_exp - bouligand4(beta, zt, dz, kh, C)))

    xopt = fmin(func, x0=np.array([zb0, zt0, C0]), args=(Phi_exp, kh, beta), full_output=True, disp=False)
    zb_opt = xopt[0][0]
    zt_opt = xopt[0][1]
    C_opt = xopt[0][2]
    misfit = xopt[1]
    return zb_opt, zt_opt, C_opt, misfit

def hfu(x):
    '''
    Transform HFU (heat flow unit) into mW/m2
    '''
    return x*41.86

def hgu(x):
    '''
    Transform HGU (heat generation unit) into µW/m3
    '''
    return x*0.418

def lachenbruch(Q0, A0, k, z, D=7500.0):
    '''
    Temperature as a fct of depth (Lachenbruch and Sass, 1977)

    Parameters
    ----------
        Q0 : heat flow             [mW/m2]
        A0 : heat production       [µW/m3]
        k  : thermal conductivity  [W/m/K]
        z  : depth                 [m]
        D  : characteristic depth  [m]

    Returns
    -------
        Temperature at z           [°C]
    '''
    return z*(Q0*1.e-3 - D*A0*1.e-6)/k + (D*D*A0*1.e-6 * (1. - np.exp(-z/D)) )/k



def lachenbruch_z(Q0, A0, k, T, D=7500.0):
    '''
    Depth as fct of temperature (infered from the model of Lachenbruch and Sass, 1977)

    Parameters
    ----------
        Q0 : heat flow             [mW/m2]
        A0 : heat production       [µW/m3]
        k  : thermal conductivity  [W/m/K]
        T  : temperature           [°C]
        D  : characteristic depth  [m]

    Returns
    -------
        Depth                      [m]
    '''
    Q = Q0*1.e-3
    A = A0*1.e-6
    z = (A*D**2 + D*(A*D - Q) * lambertw(-A*D*np.exp((-A*D**2 + k*T)/(D*(A*D - Q)))/(A*D - Q)) - k*T)/(A*D - Q)
    if np.isreal(z):
        return z.real
    else:
        raise ValueError('Complex depth returned')

def find_zb_lach(T, Q0, A0, k, z1, z2, D=7500.0):
    '''
    Find depth for a given temperature using eq of Lachenbruch and Sass (1977)

    Parameters
    ----------
        T  : temperature           [K]
        Q0 : heat flow             [mW/m2]
        A0 : heat production       [µW/m3]
        k  : thermal conductivity  [W/m/K]
        z1 : lower bound           [m]
        z2 : upper bound           [m]
        D  : characteristic depth  [m]

    Returns
    -------
        optimal depth
    '''
    def func(z, T, Q0, A0, k, D):
        return np.abs(T - lachenbruch(Q0, A0, k, z, D))

    z_opt = fminbound(func, z1, z2, args=(T, Q0, A0, k, D), disp=0)
    return z_opt

def find_D_lach(T, z, Q0, A0, k, D1, D2):
    '''
    Find characteristic depth for a given temperature - depth pair

    Parameters
    ----------
        T  : temperature           [K]
        z  : depth                 [m]
        Q0 : heat flow             [mW/m2]
        A0 : heat production       [µW/m3]
        k  : thermal conductivity  [W/m/K]
        D1 : lower bound           [m]
        D2 : upper bound           [m]

    Returns
    -------
        optimal depth
    '''
    def func(D, T, Q0, A0, k, z):
        return np.abs(T - lachenbruch(Q0, A0, k, z, D))

    D_opt = fminbound(func, D1, D2, args=(T, Q0, A0, k, z), disp=0)
    return D_opt


def find_zb_okubo(S, k, k_cut):
    ind = k>k_cut
    x = k[ind]
    y = S[ind]
    A = np.vstack([x, np.ones(len(x))]).T
    zt, c = np.linalg.lstsq(A, y)[0]

    ind = np.logical_not(ind)

    x = k[ind]
    G = np.log(np.exp(S[ind])/(x*x))
    y = G
    A = np.vstack([x, np.ones(len(x))]).T
    zo, c = np.linalg.lstsq(A, y)[0]
    zo = -zo;
    zt = -zt
    return 2*zo - zt

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    kh = np.logspace(-5.0, 0.00477121254)

    zt = 1.0
    dz = 20.0
    beta = 3.0
    C = 5.0

    # noise free test
    t1 = time.process_time()
    for n in range(100):
        Phi_exp = bouligand4(beta, zt, dz, kh, C)
    t2 = time.process_time()
    print('Time : ',str(t2-t1))
    zb = zt + dz
    beta_opt, fopt = find_beta(zb, Phi_exp, kh, beta0=1.5, C=C)
    print(beta_opt, fopt)

    beta_opt, zt_opt, fopt = find_beta_zt(zb, Phi_exp, kh, beta0=1.5, zt0=0.5, C=C)
    print(beta_opt, zt_opt, fopt)

    zb_opt, fopt = find_zb(Phi_exp, kh, beta_opt, zt_opt, zb0=15, C=C)
    print('zb_opt = '+str(zb_opt))


    # 5% noise
    Phi_exp += 0.05 * np.random.rand(Phi_exp.shape[0]) * Phi_exp
    beta_opt, fopt = find_beta(zb, Phi_exp, kh, beta0=1.5, C=C)
    print(beta_opt, fopt)
    beta_opt, zt_opt, fopt = find_beta_zt(zb, Phi_exp, kh, beta0=1.5, zt0=0.5, C=C)
    print(beta_opt, zt_opt, fopt)

    zb_opt, fopt = find_zb(Phi_exp, kh, beta_opt, zt_opt, C=5.0, zb0=15)
    print('zb_opt = '+str(zb_opt))

    zb = find_zb_okubo(Phi_exp, kh/(2*np.pi), 0.05)
    print(zb)


    show_plots = False

    if show_plots:
        plt.subplot(1,3,1)
        for v in np.arange(0.0, 2.5, 0.5):
            plt.semilogx(kh, bouligand4(beta, v, dz, kh, C),'k')
            plt.hold(True)
        plt.xlim(0.001,3)

        C = -9.0
        plt.subplot(1,3,2)
        for v in [10.0, 20.0, 50.0, 100.0, 200.0]:
            plt.semilogx(kh, bouligand4(beta, zt, v, kh, C),'k')
            plt.hold(True)
        plt.semilogx(kh, C-2.0*kh*zt - (beta-1.0)*np.log(kh),'r')
        plt.xlim(0.001,3)

        plt.subplot(1,3,3)
        for v in np.arange(0.0, 5.0):
            plt.semilogx(kh, bouligand4(v, zt, dz, kh, C),'k')
            plt.hold(True)
        plt.semilogx(kh, C-2.0*kh*zt + 2.0*np.log(1.0- np.exp(-kh*dz)),'r')
        plt.xlim(0.001,3)

        plt.show()


    T = 580.0
    Q0 = 20.0
    A0 = 0.5
    k = 2.5
    z0 = 20000.0

    z = find_zb_lach(T, Q0, A0, k, 10000.0, 100000.0)
    z1 = lachenbruch_z(Q0, A0, k, T)
    print('Lach: '+str(z)+'      '+str(z1))

    Tz = lachenbruch(Q0, A0, k, z)
    Tz1 = lachenbruch(Q0, A0, k, z1)
    print(T-Tz, T-Tz1)

    testFFTMA = False
    testSpec = True

    if testFFTMA:
        grid = Grid2d('')
        grid.ncol = 1024
        grid.nrow = 2048
        grid.dx = 0.5
        grid.dy = 0.5

        cm = [geostat.CovarianceNugget(0.2),
              geostat.CovarianceSpherical(np.array([250.0,200.0]), np.array([0]), 2.5)]

        G = grid.preFFTMA(cm)

        Z = grid.FFTMA()
        plt.matshow(Z.T)
        plt.show()

    if testSpec:
        g = Grid2d('+proj=lcc +lat_1=49 +lat_2=77 +lat_0=63 +lon_0=-92 +x_0=0 +y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs')
        g.readnc('NAmag/Qc_lcc_k.nc')

        S, k, E2, flag = g.getRadialSpectrum(606000.0, -1963000.0, 500000.0, tukey, detrend=1)

        S2, k2, E22, flag = g.getRadialSpectrum(1606000.0, -1963000.0, 500000.0, tukey)


        beta1,C1, misfit = find_beta_C(dz+zt, S, k, 3.0, 25.0)

        Phi_exp = bouligand4(beta1, zt, dz, k, C1)


        plt.semilogx(k, S, k2, S2, k, Phi_exp)
        plt.legend(('1','2','3'))
        plt.show()


#    g = Grid2d('+proj=lcc +lat_1=49 +lat_2=77 +lat_0=63 +lon_0=-92 +x_0=0 +y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs')
#    g.readnc('NAmag/Qc_lcc_k.nc')
#
#    plt.figure()
#    plt.imshow(g.data, clim=[-500.0, 600])
#
#    plt.show()
