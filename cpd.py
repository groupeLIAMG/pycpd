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
import time

import cartopy.crs as ccrs
import netCDF4
import numpy as np
import pyproj
import spectrum
from osgeo import gdal
from scipy.linalg import lstsq
from scipy.optimize import fmin, fminbound, fmin_cobyla, least_squares
from scipy.signal.windows import tukey, hann, dpss
from scipy.special import gamma, kv, lambertw

import geostat
import radon
from mem import lim_malik

dz_lb = 5.0
dz_ub = 80.0
zt_lb = 0.0
zt_ub = 5.0
beta_lb = 1.5
beta_ub = 5.8
C_lb = -np.inf
C_ub = np.inf

xtol_fmin = 0.0001
ftol_fmin = 0.0001
maxiter_fmin = None
maxfun_fmin = None

xtol_ls = 1e-8
ftol_ls = 1e-8
gtol_ls = 1e-8
max_nfev_ls = None


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Polygon:
    def __init__(self):
        self.pts = []

    def add_point(self, p):
        self.pts.append(p)

    def inside(self, p):
        test = False

        j = len(self.pts) - 1
        for i in range(j + 1):
            if ((self.pts[i].y >= p.y) != (self.pts[j].y >= p.y)) and (
                    p.x <= (self.pts[j].x - self.pts[i].x) * (p.y - self.pts[i].y) / (self.pts[j].y - self.pts[i].y) +
                    self.pts[i].x):
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
        self.Q0 = []  # mW/m2
        self.k = []  # W/m/K
        self.k_sim = []  # W/m/K
        self.A = []  # W/m3 x 1e-6
        self.A_sim = []  # W/m3 x 1e-6
        self.zb_sim = []  # m
        self.S_r = []  # Radial spectrum
        self.k_r = []  # wavenumber of Spectrum
        self.std_r = []  # standard deviation of spectrum samples in radial bin
        self.ns_r = []  # number of spectrum samples in radial bin
        self.pad = False
        self.beta_sim = []
        self.zt_sim = []
        self.C = []
        self.offshore = False
        self.zb_sat = None
        self.update_proj(proj4string)

    def __eq__(self, rhs):  # to search in lists of Forage
        if rhs is None:
            return False
        else:
            return self.is_close(rhs.lat, rhs.lon)

    def __str__(self):
        val = 'Site Name: ' + self.site_name + '\n'
        val += 'Latitude: {0:8.6f}\nLongitude: {1:8.6f}\n'.format(self.lat, self.lon)
        val += 'Q0: ' + str(self.Q0) + '\n'
        val += 'k : ' + str(self.k) + '\n'
        val += 'A : ' + str(self.A) + '\n'
        return val

    def is_close(self, lat, lon, tol=0.00001):
        return abs(self.lat - lat) < tol and abs(self.lon - lon) < tol

    def update_proj(self, proj4string):
        if proj4string is not None:
            lcc = pyproj.Proj(proj4string)
            self.x, self.y = lcc(self.lon, self.lat)


def proj4string2cartopy(proj4string):
    """
    Function to build a cartopy projection from a proj4 string

    .. warning:: This is far from complete and was tested only with the Lambert Conformal projection

    Parameters
    ----------
    proj4string : string
        string describing projection with proj4 syntax

    Returns
    -------
    proj : cartopy.crs.Projection
        Projection to be used with cartopy

    """
    params = {}
    tmp = proj4string.split('+')
    for t in tmp:
        if len(t) > 0:
            t2 = t.strip().split('=')
            if len(t2) > 1:
                params[t2[0]] = t2[1]

    # build Globe first
    datum = None
    ellipse = 'WGS84'
    semimajor_axis = None
    semiminor_axis = None
    flattening = None
    inverse_flattening = None
    towgs84 = None
    nadgrids = None

    if 'ellps' in params:
        ellipse = params['ellps']
    if 'datum' in params:
        datum = params['datum']
    if 'a' in params:
        semimajor_axis = float(params['a'])
    if 'b' in params:
        semiminor_axis = float(params['b'])
    if 'nadgrids' in params:
        nadgrids = params['nadgrids']

    globe = ccrs.Globe(datum=datum, ellipse=ellipse, semimajor_axis=semimajor_axis, semiminor_axis=semiminor_axis,
                       flattening=flattening, inverse_flattening=inverse_flattening, towgs84=towgs84, nadgrids=nadgrids)

    if params['proj'] == 'lcc':
        central_longitude = -96.0
        central_latitude = 39.0
        false_easting = 0.0
        false_northing = 0.0
        secant_latitudes = None
        standard_parallels = None
        globe = globe
        cutoff = -30

        if 'lon_0' in params:
            central_longitude = float(params['lon_0'])
        if 'lat_0' in params:
            central_latitude = float(params['lat_0'])
        if 'x_0' in params:
            false_easting = float(params['x_0'])
        if 'y_0' in params:
            false_northing = float(params['y_0'])
        if 'lat_1' in params and 'lat_2' in params:
            standard_parallels = (float(params['lat_1']), float(params['lat_2']))

        proj = ccrs.LambertConformal(central_longitude=central_longitude, central_latitude=central_latitude,
                                     false_easting=false_easting, false_northing=false_northing,
                                     secant_latitudes=secant_latitudes, standard_parallels=standard_parallels,
                                     globe=globe, cutoff=cutoff)
    elif params['proj'] == 'tmerc':
        central_longitude = 0.0
        central_latitude = 0.0
        false_easting = 0.0
        false_northing = 0.0
        scale_factor = 1.0

        if 'lon_0' in params:
            central_longitude = float(params['lon_0'])
        if 'lat_0' in params:
            central_latitude = float(params['lat_0'])
        if 'x_0' in params:
            false_easting = float(params['x_0'])
        if 'y_0' in params:
            false_northing = float(params['y_0'])
        if 'k_0' in params:
            scale_factor = float(params['k_0'])

        proj = ccrs.TransverseMercator(central_longitude=central_longitude, central_latitude=central_latitude,
                                       false_easting=false_easting, false_northing=false_northing,
                                       scale_factor=scale_factor, globe=globe)
    else:
        raise NotImplementedError('Projection ' + params['proj'] + ' not yet implemented')

    return proj


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
        self.proj4string = proj4string

    def inside(self, pt, buffer=0.0):
        return (self.xwest + buffer) <= pt.x <= (self.xeast - buffer) and (self.ysouth + buffer) <= pt.y <= (
                self.ynorth - buffer)

    def as_2col(self):
        x = self.xwest + self.dx * np.arange(self.ncol)
        y = self.ysouth + self.dy * np.arange(self.nrow)
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        return np.hstack((np.kron(np.ones((self.nrow, 1)), x), np.kron(y, np.ones((self.ncol, 1)))))

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
            self.xeast = self.xwest + self.dx * (self.ncol - 1)
            self.ysouth = struct.unpack('>f', ifile.read(4))[0]
            self.dy = struct.unpack('>f', ifile.read(4))[0]
            self.ynorth = self.ysouth + self.dy * (self.nrow - 1)
            ifile.seek(24, 1)
            pos = ifile.tell()
            ifile.seek(0, 2)
            pos2 = ifile.tell()
            nd = int((pos2 - pos) / 4)
            ifile.seek(pos, 0)
            A = np.array(struct.unpack('>' + str(nd) + 'f', ifile.read(nd * 4)))
            A[A > 1.e+31] = np.nan
            A[A < -1.e+31] = np.nan
            nmiss = (self.ncol + 3) * self.nrow - A.size
            if nmiss > 0:
                A = np.append(A, np.zeros((nmiss,)))
            A = A.reshape(self.nrow, self.ncol + 3)
            self.data = A[:, :-3]

    def ncdump(self, fname='test.nc', complevel=4):
        dataset = netCDF4.Dataset(fname, 'w', format='NETCDF4')

        dataset.createDimension('x', self.ncol)
        dataset.createDimension('y', self.nrow)

        varx = dataset.createVariable('x', np.float64, ('x'))
        vary = dataset.createVariable('y', np.float64, ('y'))
        varz = dataset.createVariable('z', np.float32, ('y', 'x'), zlib=True, complevel=complevel, fill_value=np.nan)
        varz.actual_range = [np.min(self.data), np.max(self.data)]

        varx[:] = self.xwest + self.dx * np.arange(self.ncol, dtype=np.float64)
        vary[:] = self.ysouth + self.dy * np.arange(self.nrow, dtype=np.float64)
        varz[:] = np.array(self.data, dtype=np.float32)
        # set range to ensure correct grid registration
        varx.actual_range = [varx[0], varx[-1]]
        vary.actual_range = [vary[0], vary[-1]]

        dataset.close()

    def read_nc(self, fname):
        try:
            dataset = netCDF4.Dataset(fname, 'r', format='NETCDF4')
        except OSError:
            raise IOError('Could not open ' + fname)

        if 'x_range' in dataset.variables:
            x = dataset.variables['x_range']
            self.xwest = x[0].data
            self.xeast = x[1].data
            y = dataset.variables['y_range']
            self.ysouth = y[0].data
            self.ynorth = y[1].data
            nd = dataset.variables['dimension']
            self.ncol = nd[0]
            self.nrow = nd[1]
            dx = dataset.variables['spacing']
            self.dx = dx[0]
            self.dy = dx[1]
            self.data = np.array(dataset.variables['z']).reshape((self.nrow, self.ncol))
            self.data = self.data[::-1, :]
        else:
            try:
                x = dataset.variables['x']
            except KeyError:
                x = dataset.variables['lon']
                self.latlon = True
            self.xwest = x[0].data
            self.xeast = x[-1].data
            self.ncol = x.size
            self.dx = x[1] - x[0]
            try:
                y = dataset.variables['y']
            except KeyError:
                y = dataset.variables['lat']
            self.ysouth = y[0].data
            self.ynorth = y[-1].data
            self.nrow = y.size
            self.dy = y[1] - y[0]
            self.data = np.array(dataset.variables['z'])
        dataset.close()

    def read_gdal(self, fname):
        raster = gdal.Open(fname)

        if self.proj4string == 'gdal':
            self.proj4string = raster.GetSpatialRef().ExportToProj4()

        geo_transform = raster.GetGeoTransform()
        self.dx = geo_transform[1]
        self.dy = -geo_transform[5]
        self.xwest = geo_transform[0]
        self.xeast = self.xwest + geo_transform[1] * raster.RasterXSize

        self.ynorth = geo_transform[3]
        self.ysouth = self.ynorth + geo_transform[5] * raster.RasterYSize

        self.ncol = raster.RasterXSize
        self.nrow = raster.RasterYSize
        self.data = raster.ReadAsArray()
        # flip up-down for cartopy
        self.data = self.data[::-1, :]

    def preFFTMA(self, covariance_models):
        """
        Compute matrix G for FFT-MA simulations

        Parameters
        ----------
        covariance_models: list of geostat.Covariance
            covariance models

        """
        small = 1.0e-6
        Nx = 2 * self.ncol
        Ny = 2 * self.nrow

        Nx2 = Nx / 2
        Ny2 = Ny / 2

        x = self.dx * np.hstack((np.arange(Nx2), np.arange(-Nx2 + 1, 1)))
        y = self.dy * np.hstack((np.arange(Ny2), np.arange(-Ny2 + 1, 1)))

        x = np.kron(x, np.ones((Ny,)))
        y = np.kron(y, np.ones((1, Nx)).T).flatten()

        d = 0
        for c in covariance_models:
            d = d + c.compute(np.vstack((x, y)).T, np.zeros((1, 2)))
        K = d.reshape(Nx, Ny)

        mk = True
        while mk:
            mk = False
            if np.min(K[0, :]) > small:
                # Enlarge grid to make sure that covariance falls to zero
                Ny = 2 * Ny
                mk = True

            if np.min(K[:, 0]) > small:
                Nx = 2 * Nx
                mk = True

            if mk:
                Nx2 = Nx / 2
                Ny2 = Ny / 2

                x = self.dx * np.hstack((np.arange(Nx2), np.arange(-Nx2 + 1, 1)))
                y = self.dy * np.hstack((np.arange(Ny2), np.arange(-Ny2 + 1, 1)))

                x = np.kron(x, np.ones((Ny,)))
                y = np.kron(y, np.ones((1, Nx)).T).flatten()

                d = 0
                for c in covariance_models:
                    d = d + c.compute(np.vstack((x, y)).T, np.zeros((1, 2)))
                K = d.reshape(Nx, Ny)

        self.G = np.sqrt(np.fft.fft2(K))

    def FFTMA(self):
        """
        Perform FFT-MA simulation using pre-computed spectral matrix

        Returns
        -------
        Z: ndarray
            simulated field of size nx x ny
        """
        if self.G is None:
            raise RuntimeError('Spectral matrix G should be precomputed')

        Nx, Ny = self.G.shape
        U = np.fft.fft2(np.random.randn(self.G.shape[0], self.G.shape[1]))

        Z = np.real(np.fft.ifft2(self.G * U))
        return Z[:self.ncol, :self.nrow].T  # transpose to have nx cols & ny rows

    def _get_subgrid(self, xc, yc, ww, detrend):

        nw = 1 + int(ww / self.dx)
        nw2 = int(nw / 2)
        ix = int((xc - self.xwest) / self.dx)
        iy = int((yc - self.ysouth) / self.dx)

        imin = ix - nw2
        imax = ix + nw2 + 1
        jmin = iy - nw2
        jmax = iy + nw2 + 1
        flag_pad = False
        if imin < 0:
            print('Warning: pt close to eastern edge, padding will be applied (' + str(-imin) + ' cols)')
            imin = 0
            flag_pad = True
        if imax > self.ncol:
            print('Warning: pt close to western edge, padding will be applied (' + str(imax - self.ncol) + ' cols)')
            imax = self.ncol
            flag_pad = True
        if jmin < 0:
            print('Warning: pt close to southern edge, padding will be applied (' + str(-jmin) + ' rows)')
            jmin = 0
            flag_pad = True
        if jmax > self.nrow:
            print('Warning: pt close to northern edge, padding will be applied (' + str(jmax - self.nrow) + ' rows)')
            jmax = self.nrow
            flag_pad = True

        data = self.data[jmin:jmax, imin:imax].copy()

        ny, nx = data.shape
        if ny < nw:
            data = np.vstack((data, np.zeros((nw - ny, nx))))  # pad with zeros
        if nx < nw:
            data = np.hstack((data, np.zeros((nw, nw - nx))))

        if detrend == 1:
            # remove linear trend
            x, y = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))

            A = np.column_stack((x.flatten(), y.flatten(), np.ones(x.size)))
            c, resid, rank, sigma = lstsq(A, data.flatten())

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

        return data, nw, flag_pad

    def get_radial_spectrum(self, xc, yc, ww, taper=hann, detrend=0, scal_fac=0.001, mem=0, memest=0, order=10,
                            kcut=0.0, cdecim=0, logspace=0, padding=0):
        """
        Compute radial spectrum for point at (xc,yc) for square window of
        width equal to ww

        Parameters
        ----------
        xc       : X coordinate of point
        yc       : Y coordinate of point
        ww       : window width
        taper    : taper window
        detrend  : control detrending of data
                   0 : data unchanged (default)
                   1 : remove best-fitting linear trend
                   2 : remove mean value
                   3 : remove median value
                   4 : remove mid value, i.e. 0.5 * (max + min)
        scal_fac  : scaling factor to get k in rad/km  (0.001 by default, for grid in m)
        mem      : If 0, calculate spectrum with FFT (the default)
                   If 1, calculate spectrum with maximum entropy estimator of Srinivasa et al. (1992),
                      IEEE Trans. Signal Processing
                   If 2, calculate spectrum with maximum entropy estimator of Lim & Malik
        memest   : estimator for method of Srinivasa et al. (1992)
                     If 0, use FFT (the default)
                     if 1, use ARMA model
        order    : order of ARMA estimator
        kcut     : wavenumber value at which to troncate high part of spectrum
                     If 0, do not troncate (the default)
        cdecim   : cascade decimation, value is maximum number of points to skip
                     (default is 0 : decimation not performed)
        logspace : compute spectrum along rings of logarithmically growing width, (produces a radial spectrum evenly
                     distributed on a log scale)
                     value is number of points in spectrum
                     (0 by default, use constant ring size)
        padding : interpolate spectrum by zero-padding the data
                     (0 by default, no padding)

        Returns
        -------
        S        : Radial spectrum
        k        : wavenumber [rad/km]
        std      : Standard deviation of samples in radial bin
        ns       : Number of samples in radial bin
        flag_pad : True if zero padding was applied
        """

        # check if in grid
        if xc < self.xwest or xc > self.xeast or yc < self.ysouth or yc > self.ynorth:
            raise ValueError('Point outside grid')

        if self.dx != self.dy:
            raise RuntimeError('Grid cell size should be equal in x and y')

        if logspace > 0 and cdecim > 0:
            raise ValueError('Parameters logspace and cdecim are mutually exclusive')

        if taper is dpss and mem != 0:
            raise ValueError('Multitaper not implemented for maximum entropy estimator')

        data, nw, flag_pad = self._get_subgrid(xc, yc, ww, detrend)

        # first dimension
        if taper is None:
            pass
        elif taper is tukey:
            ht0 = taper(data.shape[0], alpha=0.05)  # Bouligand uses 5% tapering
        elif taper is dpss:
            NW = 2.5
            kmax = int(2*NW - 0.999999)
            ht0, r0 = taper(data.shape[0], NW=NW, Kmax=kmax, norm='subsample', return_ratios=True)
        else:
            ht0 = taper(data.shape[0])

        # second dimension
        if taper is None:
            pass
        elif taper is tukey:
            ht1 = taper(data.shape[1], alpha=0.05)
        elif taper is dpss:
            NW = 2.5
            kmax = int(2 * NW - 0.999999)
            ht1, r1 = taper(data.shape[1], NW=NW, Kmax=kmax, norm='subsample', return_ratios=True)
        else:
            ht1 = taper(data.shape[1])

        if taper is None:
            taper_val = 1.0
        elif taper is dpss:
            taper_val = []
            weights = []
            for n0 in range(kmax):
                for n1 in range(kmax):
                    taper_val.append(np.outer(ht0[n0, :], ht1[n1, :]))
                    weights.append(r0[n0] * r1[n1])
        else:
            taper_val = np.outer(ht0, ht1)

        nfft = data.shape
        if padding > 0:
            nfft = (nfft[0] * (padding + 1), nfft[1] * (padding + 1))

        dx = self.dx * scal_fac  # from m to km
        dk0 = 2.0 * np.pi / (nw - 1) / dx  # before padding
        dk = 2.0 * np.pi / (nfft[0] - 1) / dx

        if mem == 0:
            if type(taper_val) is list:
                TF2D = weights[0] * np.abs(np.fft.fft2(data * taper_val[0], s=nfft))
                for n in range(1, len(taper_val)):
                    TF2D += weights[n] * np.abs(np.fft.fft2(data * taper_val[n], s=nfft))
                TF2D /= np.sum(np.array(weights))
            else:
                TF2D = np.abs(np.fft.fft2(data * taper_val, s=nfft))
            TF2D = np.fft.fftshift(TF2D)

            if logspace == 0:
                kbins = np.arange(dk0, dk0 * nw / 2, dk0)
            else:
                kbins = np.logspace(np.log10(dk0), np.log10(dk0 * nw / 2), logspace)

            nbins = kbins.size - 1
            S = np.zeros((nbins,))
            k = np.zeros((nbins,))
            std = np.zeros((nbins,))
            ns = np.zeros((nbins,))

            i0 = int((nfft[0] - 1) / 2)
            iy, ix = np.meshgrid(np.arange(nfft[0]), np.arange(nfft[0]))
            kk = np.sqrt(((ix - i0) * dk) ** 2 + ((iy - i0) * dk) ** 2)

            for n in range(nbins):
                ind = np.logical_and(kk >= kbins[n], kk < kbins[n + 1])
                rr = 2.0 * np.log(TF2D[ind])
                S[n] = np.mean(rr)
                k[n] = np.mean(kk[ind])
                std[n] = np.std(rr)
                ns[n] = rr.size

        elif mem == 1:
            # method of Srinivasa et al. 1992
            k = np.arange(dk, dk * nw / 2, dk)
            S = np.zeros((k.size,))
            std = np.zeros((k.size,))

            theta = np.pi * np.arange(0.0, 180.0, 5.0) / 180.0
            sinogram = radon.radon2d(data, theta)
            SS = np.zeros((theta.size, k.size))
            ns = np.zeros((k.size,)) + theta.size

            # apply taper on individual directions
            if taper is None:
                taper_val = 1.0
            elif taper is tukey:
                taper_val = taper(sinogram.shape[0], alpha=0.05)
            else:
                taper_val = taper(sinogram.shape[0])

            for n in range(theta.size):
                if memest == 0:
                    PSD = np.abs(np.fft.fft(taper_val * sinogram[:, n], n=1 + 2 * k.size))
                else:
                    a, b, rho = spectrum.arma_estimate(taper_val * sinogram[:, n], order, order, 2 * order)
                    PSD = spectrum.arma2psd(A=a, B=b, NFFT=1 + 2 * k.size)
                #                 AR, P, kk = spectrum.arburg(taper_val * sinogram[:,n], order)
                #                 PSD = spectrum.arma2psd(AR, NFFT=1+2*k.size)
                SS[n, :] = 2.0 * np.log(PSD[1:k.size + 1])

            S = np.mean(SS, axis=0)
            std = np.std(SS, axis=0)

        elif mem == 2:
            Stmp = np.abs(lim_malik(data * taper_val))

            kbins = np.arange(dk, dk * nw / 2, dk)
            nbins = kbins.size - 1
            S = np.zeros((nbins,))
            k = np.zeros((nbins,))
            std = np.zeros((nbins,))
            ns = np.zeros((nbins,))

            i0 = int((nw - 1) / 2)
            iy, ix = np.meshgrid(np.arange(nw), np.arange(nw))
            kk = np.sqrt(((ix - i0) * dk) ** 2 + ((iy - i0) * dk) ** 2)

            for n in range(nbins):
                ind = np.logical_and(kk >= kbins[n], kk < kbins[n + 1])
                rr = 2.0 * np.log(Stmp[ind])
                S[n] = np.mean(rr)
                k[n] = np.mean(kk[ind])
                std[n] = np.std(rr)
                ns[n] = rr.size

        else:
            raise ValueError('Method undefined')

        if kcut != 0.0:
            ind = k < kcut
            S = S[ind]
            k = k[ind]
            std = std[ind]
            ns = ns[ind]

        if cdecim > 0:
            N = k.size
            ind = np.zeros(N, np.bool_)
            # \sum_{s=0}^{s_{max}} n(s+1) & < N \\
            # i_0 + \sum_{m=0}^{s_{max}} n(s+1) & = N
            n = int(N / np.sum(np.arange(cdecim + 2)))
            i0 = N - np.sum(n * (np.arange(cdecim + 1) + 1))
            ind[:i0] = True
            for s in range(cdecim + 1):
                i0 += n * s
                for nn in range(n):
                    ind[i0 + (nn + 1) * (s + 1) - 1] = True
            S = S[ind]
            k = k[ind]
            std = std[ind]
            ns = ns[ind]

        ind = np.isfinite(k)
        S = S[ind]
        k = k[ind]
        std = std[ind]
        ns = ns[ind]

        return S, k, std, ns, flag_pad

    def get_azimuthal_spectrum(self, xc, yc, ww, taper=hann, detrend=0, scal_fac=0.001, dtheta=5.0, memest=0,
                               order=10):
        """
        Compute radial spectrum for point at (xc,yc) for square window of
        width equal to ww

        Parameters
        ----------
        xc       : X coordinate of point
        yc       : Y coordinate of point
        ww       : window width
        taper    : taper window
        detrend  : control detrending of data
                   0 : data unchanged (default)
                   1 : remove best-fitting linear trend
                   2 : remove mean value
                   3 : remove median value
                   4 : remove mid value, i.e. 0.5 * (max + min)
        scal_fac  : scaling factor to get k in rad/km  (0.001 by default, for grid in m)
        dtheta   : angle increment in degrees
        memest   : estimator for method of Srinivasa et al. (1992)
                     If 0, use FFT (the default)
                     if 1, use ARMA model
        order    : order of ARMA estimator


        Returns
        -------
        S       : Azimuthal spectrum
        k       : wavenumber [rad/km]
        theta   : angles [°]
        flagPad : True if zero padding was applied
        """

        # check if in grid
        if xc < self.xwest or xc > self.xeast or yc < self.ysouth or yc > self.ynorth:
            raise ValueError('Point outside grid')

        if self.dx != self.dy:
            raise RuntimeError('Grid cell size should be equal in x and y')

        data, nw, flagPad = self._get_subgrid(xc, yc, ww, detrend)

        dx = self.dx * scal_fac  # from m to km
        dk = 2.0 * np.pi / (nw - 1) / dx

        # method of Srinivasa et al. 1992
        k = np.arange(dk, dk * nw / 2, dk)

        theta = np.arange(0.0, 180.0, dtheta)
        sinogram = radon.radon2d(data, np.pi * theta / 180.0)
        SS = np.zeros((theta.size, k.size))

        taper_val = np.ones(data.shape)
        if taper is tukey:
            taper_val = taper(sinogram.shape[0], alpha=0.05)
        else:
            taper_val = taper(sinogram.shape[0])

        for n in range(theta.size):
            if memest == 0:
                PSD = np.abs(np.fft.fft(taper_val * sinogram[:, n], n=1 + 2 * k.size))
            else:
                a, b, rho = spectrum.arma_estimate(taper_val * sinogram[:, n], order, order, 2 * order)
                PSD = spectrum.arma2psd(A=a, B=b, NFFT=1 + 2 * k.size)
            SS[n, :] = 2.0 * np.log(PSD[1:k.size + 1])

        return SS, k, theta, flagPad


def bouligand4(beta, zt, dz, kh, C=0.0):
    """
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
    """
    khdz = kh * dz
    coshkhdz = np.cosh(khdz)

    phi1d = C - 2.0 * kh * zt - (beta - 1.0) * np.log(kh) - khdz
    A = np.sqrt(np.pi) / gamma(1.0 + 0.5 * beta) * (
            0.5 * coshkhdz * gamma(0.5 * (1.0 + beta)) - kv((-0.5 * (1.0 + beta)), khdz) * np.power(0.5 * khdz, (
            0.5 * (1.0 + beta))))
    phi1d += np.log(A)
    return phi1d


def find_beta(dz, Phi_exp, kh, beta0, zt=1.0, C=0, wlf=False, method='fmin', lb=[], ub=[]):
    """
    Find fractal parameter beta for a given radial spectrum

    Parameters
    ----------
        dz      : thichness of magnetic layer
        Phi_exp : Spectrum values for kh
        kh      : norm of the wave number in the horizontal plane
        beta0   : starting value
        zt      : depth of top of magnetic layer
        C       : field constant (Maus et al., 1997)
        wlf     : apply low frequency weighting
        method  : 'fmin' -> simplex method
                  'ls' -> least-squares
        lb      : lower bounds for least-squares (beta)
        ub      : upper bounds for least-squares (beta)

    Returns
    -------
        beta, Normalized RMS misfit
    """
    if not np.isscalar(wlf):
        # wlf must be array of size Phi_exp
        w = 1.0 / wlf
    elif wlf:
        w = np.linspace(1.5, 0.5, Phi_exp.size)
    else:
        w = 1.0

    # define function to minimize
    def func(beta, dz, Phi_exp, zt, kh, C):
        return np.sqrt(1.0 / Phi_exp.size * np.sum((w * (Phi_exp - bouligand4(beta, zt, dz, kh, C)) ** 2)))

    if method == 'fmin':
        xopt = fmin(func, x0=beta0, args=(dz, Phi_exp, zt, kh, C), full_output=True, disp=False, xtol=xtol_fmin,
                    ftol=ftol_fmin, maxiter=maxiter_fmin, maxfun=maxfun_fmin)
        beta_opt = xopt[0][0]
        misfit = xopt[1]

    elif method == 'ls':
        if len(lb) == 0:
            lb = np.array([beta_lb])
        if len(ub) == 0:
            ub = np.array([beta_ub])

        res = least_squares(func, x0=beta0, jac='3-point', bounds=(lb, ub), args=(dz, Phi_exp, zt, kh, C), xtol=xtol_ls,
                            ftol=ftol_ls, gtol=gtol_ls, max_nfev=max_nfev_ls)
        beta_opt = res.x[0]
        misfit = res.cost

    else:
        raise ValueError('Method undefined')

    return beta_opt, misfit


def find_zt(dz, Phi_exp, kh, beta, zt0, C=0, wlf=False, method='fmin', lb=[], ub=[]):
    """
    Find depth of top of magnetic layer for a given radial spectrum

    Parameters
    ----------
        dz      : thichness of magnetic layer
        Phi_exp : Spectrum values for kh
        kh      : norm of the wave number in the horizontal plane
        beta    : fractal paremeter
        zt0     : starting value
        C       : field constant (Maus et al., 1997)
        wlf     : apply low frequency weighting
        method  : 'fmin' -> simplex method
                  'ls' -> least-squares
        lb      : lower bounds for least-squares (zt)
        ub      : upper bounds for least-squares (zt)

    Returns
    -------
       zt, Normalized RMS misfit
    """
    if not np.isscalar(wlf):
        # wlf must be array of size Phi_exp
        w = 1.0 / wlf
    elif wlf:
        w = np.linspace(1.5, 0.5, Phi_exp.size)
    else:
        w = 1.0

    # define function to minimize
    def func(zt, dz, Phi_exp, beta, kh, C):
        return np.sqrt(1.0 / Phi_exp.size * np.sum((w * (Phi_exp - bouligand4(beta, zt, dz, kh, C)) ** 2)))

    if method == 'fmin':
        xopt = fmin(func, x0=zt0, args=(dz, Phi_exp, beta, kh, C), full_output=True, disp=False, xtol=xtol_fmin,
                    ftol=ftol_fmin, maxiter=maxiter_fmin, maxfun=maxfun_fmin)
        zt_opt = xopt[0][0]
        misfit = xopt[1]

    elif method == 'ls':
        if len(lb) == 0:
            lb = np.array([zt_lb])
        if len(ub) == 0:
            ub = np.array([zt_ub])

        res = least_squares(func, x0=zt0, jac='3-point', bounds=(lb, ub), args=(dz, Phi_exp, beta, kh, C), xtol=xtol_ls,
                            ftol=ftol_ls, gtol=gtol_ls, max_nfev=max_nfev_ls)
        zt_opt = res.x[0]
        misfit = res.cost

    else:
        raise ValueError('Method undefined')

    return zt_opt, misfit


def find_dz(dz0, Phi_exp, kh, beta, zt, C, wlf=False, method='fmin', lb=[], ub=[]):
    """
    Find thickness of magnetic layer for a given radial spectrum

    Parameters
    ----------
        dz0     : starting value
        Phi_exp : Spectrum values for kh
        kh      : norm of the wave number in the horizontal plane
        beta    : fractal paremeter
        zt      : depth of top of magnetic layer
        C       : field constant (Maus et al., 1997)
        wlf     : apply low frequency weighting
        method  : 'fmin' -> simplex method
                  'ls' -> least-squares
        lb      : lower bounds for least-squares (dz)
        ub      : upper bounds for least-squares (dz)

    Returns
    -------
       dz, Normalized RMS misfit
    """
    if not np.isscalar(wlf):
        # wlf must be array of size Phi_exp
        w = 1.0 / wlf
    elif wlf:
        w = np.linspace(1.5, 0.5, Phi_exp.size)
    else:
        w = 1.0

    # define function to minimize
    def func(dz, zt, Phi_exp, beta, kh, C):
        return np.sqrt(1.0 / Phi_exp.size * np.sum((w * (Phi_exp - bouligand4(beta, zt, dz, kh, C)) ** 2)))

    if method == 'fmin':
        xopt = fmin(func, x0=dz0, args=(zt, Phi_exp, beta, kh, C), full_output=True, disp=False, xtol=xtol_fmin,
                    ftol=ftol_fmin, maxiter=maxiter_fmin, maxfun=maxfun_fmin)
        dz_opt = xopt[0][0]
        misfit = xopt[1]

    elif method == 'ls':
        if len(lb) == 0:
            lb = np.array([dz_lb])
        if len(ub) == 0:
            ub = np.array([dz_ub])

        res = least_squares(func, x0=dz0, jac='3-point', bounds=(lb, ub), args=(zt, Phi_exp, beta, kh, C), xtol=xtol_ls,
                            ftol=ftol_ls, gtol=gtol_ls, max_nfev=max_nfev_ls)
        dz_opt = res.x[0]
        misfit = res.cost

    else:
        raise ValueError('Method undefined')

    return dz_opt, misfit


def find_C(dz, Phi_exp, kh, beta, zt, C0, wlf=False, method='fmin', lb=[], ub=[]):
    """
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
        method  : 'fmin' -> simplex method
                  'ls' -> least-squares
        lb      : lower bounds for least-squares (C)
        ub      : upper bounds for least-squares (C)

    Returns
    -------
       C, Normalized RMS misfit
    """
    if not np.isscalar(wlf):
        # wlf must be array of size Phi_exp
        w = 1.0 / wlf
    elif wlf:
        w = np.linspace(1.5, 0.5, Phi_exp.size)
    else:
        w = 1.0

    # define function to minimize
    def func(C, zt, Phi_exp, beta, kh, dz):
        return np.sqrt(1.0 / Phi_exp.size * np.sum((w * (Phi_exp - bouligand4(beta, zt, dz, kh, C)) ** 2)))

    if method == 'fmin':
        xopt = fmin(func, x0=C0, args=(zt, Phi_exp, beta, kh, dz), full_output=True, disp=False, xtol=xtol_fmin,
                    ftol=ftol_fmin, maxiter=maxiter_fmin, maxfun=maxfun_fmin)
        C_opt = xopt[0][0]
        misfit = xopt[1]

    elif method == 'ls':
        if len(lb) == 0:
            lb = np.array([C_lb])
        if len(ub) == 0:
            ub = np.array([C_ub])

        res = least_squares(func, x0=C0, jac='3-point', bounds=(lb, ub), args=(zt, Phi_exp, beta, kh, dz), xtol=xtol_ls,
                            ftol=ftol_ls, gtol=gtol_ls, max_nfev=max_nfev_ls)
        C_opt = res.x[0]
        misfit = res.cost

    else:
        raise ValueError('Method undefined')

    return C_opt, misfit


def find_beta_zt_dz_C(Phi_exp, kh, beta0, zt0, dz0, C0, wlf=False, method='fmin', lb=[], ub=[]):
    """
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
        method  : 'fmin' -> simplex method
                  'ls' -> least-squares
        lb      : lower bounds for least-squares (beta, zt, dz, C)
        ub      : upper bounds for least-squares (beta, zt, dz, C)

    Returns
    -------
        beta, zt, dz, C, Normalized RMS misfit
    """
    if not np.isscalar(wlf):
        # wlf must be array of size Phi_exp
        w = 1.0 / wlf
    elif wlf:
        w = np.linspace(1.5, 0.5, Phi_exp.size)
    else:
        w = 1.0

    # define function to minimize
    def func(x, Phi_exp, kh):
        beta = x[0]
        zt = x[1]
        dz = x[2]
        C = x[3]
        return np.sqrt(1.0 / Phi_exp.size * np.sum((w * (Phi_exp - bouligand4(beta, zt, dz, kh, C)) ** 2)))

    if method == 'fmin':
        xopt = fmin(func, x0=np.array([beta0, zt0, dz0, C0]), args=(Phi_exp, kh), full_output=True, disp=False,
                    xtol=xtol_fmin, ftol=ftol_fmin, maxiter=maxiter_fmin, maxfun=maxfun_fmin)
        beta_opt = xopt[0][0]
        zt_opt = xopt[0][1]
        dz_opt = xopt[0][2]
        C_opt = xopt[0][3]
        misfit = xopt[1]

    elif method == 'ls':
        if len(lb) == 0:
            lb = np.array([beta_lb, zt_lb, dz_lb, C_lb])
        if len(ub) == 0:
            ub = np.array([beta_ub, zt_ub, dz_ub, C_ub])

        res = least_squares(func, x0=np.array([beta0, zt0, dz0, C0]), jac='3-point', bounds=(lb, ub),
                            args=(Phi_exp, kh), xtol=xtol_ls, ftol=ftol_ls, gtol=gtol_ls, max_nfev=max_nfev_ls)
        beta_opt = res.x[0]
        zt_opt = res.x[1]
        dz_opt = res.x[2]
        C_opt = res.x[3]
        misfit = res.cost

    else:
        raise ValueError('Method undefined')

    return beta_opt, zt_opt, dz_opt, C_opt, misfit


def find_beta_zt_C(Phi_exp, kh, beta0, zt0, C0, dz, wlf=False, method='fmin', lb=[], ub=[]):
    """
    Find fractal model parameters, depth of top of magnetic layer and
    constant C for a given radial spectrum and depth to bottom value

    Parameters
    ----------
        Phi_exp : Spectrum values for kh
        kh      : norm of the wave number in the horizontal plane
        beta0   : starting value of beta
        zt0     : starting value of depth to top of magnetic layer
        C0      : starting value of field constant (Maus et al., 1997)
        dz      : thickness of magnetic layer
        wlf     : apply low frequency weighting
        method  : 'fmin' -> simplex method
                  'ls' -> least-squares
        lb      : lower bounds for least-squares (beta, zt, C)
        ub      : upper bounds for least-squares (beta, zt, C)

    Returns
    -------
        beta, zt, C, Normalized RMS misfit
    """
    if not np.isscalar(wlf):
        # wlf must be array of size Phi_exp
        w = 1.0 / wlf
    elif wlf:
        w = np.linspace(1.5, 0.5, Phi_exp.size)
    else:
        w = 1.0
    # define function to minimize
    iN = 1.0 / Phi_exp.size

    def func(x, Phi_exp, kh, dz):
        beta = x[0]
        zt = x[1]
        C = x[2]
        return np.sqrt(iN * np.sum((w * (Phi_exp - bouligand4(beta, zt, dz, kh, C)) ** 2)))

    if method == 'fmin':
        xopt = fmin(func, x0=np.array([beta0, zt0, C0]), args=(Phi_exp, kh, dz), full_output=True, disp=False,
                    xtol=xtol_fmin, ftol=ftol_fmin, maxiter=maxiter_fmin, maxfun=maxfun_fmin)
        beta_opt = xopt[0][0]
        zt_opt = xopt[0][1]
        C_opt = xopt[0][2]
        misfit = xopt[1]

    elif method == 'ls':
        if len(lb) == 0:
            lb = np.array([beta_lb, zt_lb, C_lb])
        if len(ub) == 0:
            ub = np.array([beta_ub, zt_ub, C_ub])

        res = least_squares(func, x0=np.array([beta0, zt0, C0]), jac='3-point', bounds=(lb, ub), args=(Phi_exp, kh, dz),
                            xtol=xtol_ls, ftol=ftol_ls, gtol=gtol_ls, max_nfev=max_nfev_ls)
        beta_opt = res.x[0]
        zt_opt = res.x[1]
        C_opt = res.x[2]
        misfit = res.cost

    elif method == '2s':
        if len(lb) == 0:
            lb = np.array([beta_lb, zt_lb, C_lb])
        if len(ub) == 0:
            ub = np.array([beta_ub, zt_ub, C_ub])

        res = least_squares(func, x0=np.array([beta0, zt0, C0]), jac='3-point', bounds=(lb, ub), args=(Phi_exp, kh, dz),
                            xtol=xtol_ls, ftol=ftol_ls, gtol=gtol_ls, max_nfev=max_nfev_ls)
        beta_opt = res.x[0]
        zt_opt = res.x[1]
        C_opt = res.x[2]

        xopt = fmin(func, x0=np.array([beta_opt, zt_opt, C_opt]), args=(Phi_exp, kh, dz), full_output=True, disp=False,
                    xtol=xtol_fmin, ftol=ftol_fmin, maxiter=maxiter_fmin, maxfun=maxfun_fmin)
        beta_opt = xopt[0][0]
        zt_opt = xopt[0][1]
        C_opt = xopt[0][2]
        misfit = xopt[1]

    elif method == '2sb':
        xopt = fmin(func, x0=np.array([beta0, zt0, C0]), args=(Phi_exp, kh, dz), full_output=True, disp=False,
                    xtol=xtol_fmin, ftol=ftol_fmin, maxiter=maxiter_fmin, maxfun=maxfun_fmin)
        beta_opt = xopt[0][0]
        zt_opt = xopt[0][1]
        C_opt = xopt[0][2]

        if len(lb) == 0:
            lb = np.array([beta_lb, zt_lb, C_lb])
        if len(ub) == 0:
            ub = np.array([beta_ub, zt_ub, C_ub])

        if beta_opt < lb[0]:
            beta_opt = lb[0]
        if beta_opt > ub[0]:
            beta_opt = ub[0]
        if zt_opt < lb[1]:
            zt_opt = lb[1]
        if zt_opt > ub[1]:
            zt_opt = ub[1]
        if C_opt < lb[2]:
            C_opt = lb[2]
        if C_opt > ub[2]:
            C_opt = ub[2]

        res = least_squares(func, x0=np.array([beta_opt, zt_opt, C_opt]), jac='3-point', bounds=(lb, ub),
                            args=(Phi_exp, kh, dz), xtol=xtol_ls, ftol=ftol_ls, gtol=gtol_ls, max_nfev=max_nfev_ls)
        beta_opt = res.x[0]
        zt_opt = res.x[1]
        C_opt = res.x[2]
        misfit = res.cost

    else:
        raise ValueError('Method undefined')

    return beta_opt, zt_opt, C_opt, misfit


def find_beta_zt_C_bound(Phi_exp, kh, beta, zt, C, zb, wlf=False):
    """
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
        beta, zt, C, Normalized RMS misfit
    """
    beta0, beta1, beta2 = beta
    zt0, zt1, zt2 = zt
    C0, C1, C2 = C

    if not np.isscalar(wlf):
        # wlf must be array of size Phi_exp
        w = 1.0 / wlf
    elif wlf:
        w = np.linspace(1.5, 0.5, Phi_exp.size)
    else:
        w = 1.0

    # define function to minimize
    def func(x, Phi_exp, kh, zb):
        beta = x[0]
        zt = x[1]
        dz = zb - zt
        C = x[2]
        return np.sqrt(1.0 / Phi_exp.size * np.sum((w * (Phi_exp - bouligand4(beta, zt, dz, kh, C)) ** 2)))

    def cons1(x):
        return beta2 - x[0], zt2 - x[1], C2 - x[2]

    def cons2(x):
        return x[1] - beta1, x[1] - zt1, x[2] - C1

    xopt = fmin_cobyla(func, x0=np.array([beta0, zt0, C0]), cons=(cons1, cons2), args=(Phi_exp, kh, zb), consargs=(),
                       disp=False)
    beta_opt = xopt[0]
    zt_opt = xopt[1]
    C_opt = xopt[2]
    misfit = func(xopt, Phi_exp, kh, zb)
    return beta_opt, zt_opt, C_opt, misfit


def find_beta_dz_C(Phi_exp, kh, beta0, dz0, C0, zt=1.0, wlf=False, method='fmin', lb=[], ub=[]):
    """
    Find fractal model parameters, depth of bottom of magnetic layer and
    constant C for a given radial spectrum

    Parameters
    ----------
        Phi_exp : Spectrum values for kh
        kh      : norm of the wave number in the horizontal plane
        beta0   : starting value of beta
        dz0     : starting value of thickness of magnetic layer
        C0      : starting value of field constant (Maus et al., 1997)
        zt      : depth of top of magnetic layer
        wlf     : apply low frequency weighting
        method  : 'fmin' -> simplex method
                  'ls' -> least-squares
        lb      : lower bounds for least-squares (beta, dz, C)
        ub      : upper bounds for least-squares (beta, dz, C)

    Returns
    -------
        beta, dz, C, Normalized RMS misfit
    """
    if not np.isscalar(wlf):
        # wlf must be array of size Phi_exp
        w = 1.0 / wlf
    elif wlf:
        w = np.linspace(1.5, 0.5, Phi_exp.size)
    else:
        w = 1.0

    # define function to minimize
    def func(x, Phi_exp, kh, zt):
        beta = x[0]
        dz = x[1]
        C = x[2]
        return np.sqrt(1.0 / Phi_exp.size * np.sum((w * (Phi_exp - bouligand4(beta, zt, dz, kh, C)) ** 2)))

    if method == 'fmin':
        xopt = fmin(func, x0=np.array([beta0, dz0, C0]), args=(Phi_exp, kh, zt), full_output=True, disp=False,
                    xtol=xtol_fmin, ftol=ftol_fmin, maxiter=maxiter_fmin, maxfun=maxfun_fmin)
        beta_opt = xopt[0][0]
        dz_opt = xopt[0][1]
        C_opt = xopt[0][2]
        misfit = xopt[1]

    elif method == 'ls':
        if len(lb) == 0:
            lb = np.array([beta_lb, dz_lb, C_lb])
        if len(ub) == 0:
            ub = np.array([beta_ub, dz_ub, C_ub])

        res = least_squares(func, x0=np.array([beta0, dz0, C0]), jac='3-point', bounds=(lb, ub), args=(Phi_exp, kh, zt),
                            xtol=xtol_ls, ftol=ftol_ls, gtol=gtol_ls, max_nfev=max_nfev_ls)
        beta_opt = res.x[0]
        dz_opt = res.x[1]
        C_opt = res.x[2]
        misfit = res.cost

    else:
        raise ValueError('Method undefined')

    return beta_opt, dz_opt, C_opt, misfit


def find_beta_dz_zt(Phi_exp, kh, beta0, dz0, zt0, C, wlf=False, method='fmin', lb=[], ub=[]):
    """
    Find fractal model parameters, depth of bottom and depth to top of magnetic
    layer for a given radial spectrum

    Parameters
    ----------
        Phi_exp : Spectrum values for kh
        kh      : norm of the wave number in the horizontal plane
        beta0   : starting value of beta
        dz0     : starting value of thickness of magnetic layer
        zt0     : starting value of depth of top of magnetic layer
        C       : field constant (Maus et al., 1997)
        wlf     : apply low frequency weighting
        method  : 'fmin' -> simplex method
                  'ls' -> least-squares
        lb      : lower bounds for least-squares (beta, dz, zt)
        ub      : upper bounds for least-squares (beta, dz, zt)

    Returns
    -------
        beta, dz, zt, Normalized RMS misfit
    """
    if not np.isscalar(wlf):
        # wlf must be array of size Phi_exp
        w = 1.0 / wlf
    elif wlf:
        w = np.linspace(1.5, 0.5, Phi_exp.size)
    else:
        w = 1.0

    # define function to minimize
    def func(x, Phi_exp, kh, C):
        beta = x[0]
        dz = x[1]
        zt = x[2]
        return np.sqrt(1.0 / Phi_exp.size * np.sum((w * (Phi_exp - bouligand4(beta, zt, dz, kh, C)) ** 2)))

    if method == 'fmin':
        xopt = fmin(func, x0=np.array([beta0, dz0, zt0]), args=(Phi_exp, kh, C), full_output=True, disp=False,
                    xtol=xtol_fmin, ftol=ftol_fmin, maxiter=maxiter_fmin, maxfun=maxfun_fmin)
        beta_opt = xopt[0][0]
        dz_opt = xopt[0][1]
        zt_opt = xopt[0][2]
        misfit = xopt[1]

    elif method == 'ls':
        if len(lb) == 0:
            lb = np.array([beta_lb, dz_lb, zt_lb])
        if len(ub) == 0:
            ub = np.array([beta_ub, dz_ub, zt_ub])

        res = least_squares(func, x0=np.array([beta0, dz0, zt0]), jac='3-point', bounds=(lb, ub), args=(Phi_exp, kh, C),
                            xtol=xtol_ls, ftol=ftol_ls, gtol=gtol_ls, max_nfev=max_nfev_ls)
        beta_opt = res.x[0]
        dz_opt = res.x[1]
        zt_opt = res.x[2]
        misfit = res.cost

    else:
        raise ValueError('Method undefined')

    return beta_opt, dz_opt, zt_opt, misfit


def find_beta_zt(dz, Phi_exp, kh, beta0, zt0, C=0, wlf=False, method='fmin', lb=[], ub=[]):
    """
    Find fractal parameter beta and depth of top of magnetic layer for a given radial spectrum

    Parameters
    ----------
        dz      : thickness of magnetic layer
        Phi_exp : Spectrum values for kh
        kh      : norm of the wave number in the horizontal plane
        beta0   : starting value
        zt0     : depth of top of magnetic layer
        C       : field constant (Maus et al., 1997)
        wlf     : apply low frequency weighting
        method  : 'fmin' -> simplex method
                  'ls' -> least-squares
        lb      : lower bounds for least-squares (beta, zt)
        ub      : upper bounds for least-squares (beta, zt)

    Returns
    -------
        beta, zt, Normalized RMS misfit
    """
    if not np.isscalar(wlf):
        # wlf must be array of size Phi_exp
        w = 1.0 / wlf
    elif wlf:
        w = np.linspace(1.5, 0.5, Phi_exp.size)
    else:
        w = 1.0

    # define function to minimize
    def func(x, dz, Phi_exp, kh, C):
        beta = x[0]
        zt = x[1]
        return np.sqrt(1.0 / Phi_exp.size * np.sum((w * (Phi_exp - bouligand4(beta, zt, dz, kh, C)) ** 2)))

    if method == 'fmin':
        xopt = fmin(func, x0=np.array([beta0, zt0]), args=(dz, Phi_exp, kh, C), full_output=True, disp=False,
                    xtol=xtol_fmin, ftol=ftol_fmin, maxiter=maxiter_fmin, maxfun=maxfun_fmin)
        beta_opt = xopt[0][0]
        zt_opt = xopt[0][1]
        misfit = xopt[1]

    elif method == 'ls':
        if len(lb) == 0:
            lb = np.array([beta_lb, zt_lb])
        if len(ub) == 0:
            ub = np.array([beta_ub, zt_ub])

        res = least_squares(func, x0=np.array([beta0, zt0]), jac='3-point', bounds=(lb, ub), args=(dz, Phi_exp, kh, C),
                            xtol=xtol_ls, ftol=ftol_ls, gtol=gtol_ls, max_nfev=max_nfev_ls)
        beta_opt = res.x[0]
        zt_opt = res.x[1]
        misfit = res.cost

    else:
        raise ValueError('Method undefined')

    return beta_opt, zt_opt, misfit


def find_beta_C(dz, Phi_exp, kh, beta0, C0, zt=1.0, wlf=False, method='fmin', lb=[], ub=[]):
    """
    Find fractal parameter beta and constant C for a given radial spectrum

    Parameters
    ----------
        dz      : thickness of magnetic layer
        Phi_exp : Spectrum values for kh
        kh      : norm of the wave number in the horizontal plane
        beta0   : starting value of beta
        C0      : starting value of C
        zt      : depth of top of magnetic layer
        wlf     : apply low frequency weighting
        method  : 'fmin' -> simplex method
                  'ls' -> least-squares
        lb      : lower bounds for least-squares (beta, C)
        ub      : upper bounds for least-squares (beta, C)

    Returns
    -------
        beta, C, Normalized RMS misfit
    """
    if not np.isscalar(wlf):
        # wlf must be array of size Phi_exp
        w = 1.0 / wlf
    elif wlf:
        w = np.linspace(1.5, 0.5, Phi_exp.size)
    else:
        w = 1.0

    # define function to minimize
    def func(x, dz, Phi_exp, kh, zt):
        beta = x[0]
        C = x[1]
        return np.sqrt(1.0 / Phi_exp.size * np.sum((w * (Phi_exp - bouligand4(beta, zt, dz, kh, C)) ** 2)))

    if method == 'fmin':
        xopt = fmin(func, x0=np.array([beta0, C0]), args=(dz, Phi_exp, kh, zt), full_output=True, disp=False,
                    xtol=xtol_fmin, ftol=ftol_fmin, maxiter=maxiter_fmin, maxfun=maxfun_fmin)
        beta_opt = xopt[0][0]
        C_opt = xopt[0][1]
        misfit = xopt[1]

    elif method == 'ls':
        if len(lb) == 0:
            lb = np.array([beta_lb, C_lb])
        if len(ub) == 0:
            ub = np.array([beta_ub, C_ub])

        res = least_squares(func, x0=np.array([beta0, C0]), jac='3-point', bounds=(lb, ub), args=(dz, Phi_exp, kh, zt),
                            xtol=xtol_ls, ftol=ftol_ls, gtol=gtol_ls, max_nfev=max_nfev_ls)
        beta_opt = res.x[0]
        C_opt = res.x[1]
        misfit = res.cost

    else:
        raise ValueError('Method undefined')

    return beta_opt, C_opt, misfit


def find_dz_zt(Phi_exp, kh, dz0, zt0, beta, C, wlf=False, method='fmin', lb=[], ub=[]):
    """
    Find fractal depth of top and thickness of magnetic layer for a given radial spectrum

    Parameters
    ----------
        Phi_exp : Spectrum values for kh
        kh      : norm of the wave number in the horizontal plane
        zt0     : starting value of depth of top of magnetic layer
        dz0     : starting value of thickness
        C       : field constant (Maus et al., 1997)
        wlf     : apply low frequency weighting
        method  : 'fmin' -> simplex method
                  'ls' -> least-squares
        lb      : lower bounds for least-squares (dz, zt)
        ub      : upper bounds for least-squares (dz, zt)

    Returns
    -------
        dz, zt, Normalized RMS misfit
    """
    if not np.isscalar(wlf):
        # wlf must be array of size Phi_exp
        w = 1.0 / wlf
    elif wlf:
        w = np.linspace(1.5, 0.5, Phi_exp.size)
    else:
        w = 1.0

    # define function to minimize
    def func(x, Phi_exp, kh, beta, C):
        dz = x[0]
        zt = x[1]
        return np.sqrt(1.0 / Phi_exp.size * np.sum((w * (Phi_exp - bouligand4(beta, zt, dz, kh, C)) ** 2)))

    if method == 'fmin':
        xopt = fmin(func, x0=np.array([dz0, zt0]), args=(Phi_exp, kh, beta, C), disp=False, full_output=True,
                    xtol=xtol_fmin, ftol=ftol_fmin, maxiter=maxiter_fmin, maxfun=maxfun_fmin)
        dz_opt = xopt[0][0]
        zt_opt = xopt[0][1]
        misfit = xopt[1]

    elif method == 'ls':
        if len(lb) == 0:
            lb = np.array([dz_lb, zt_lb])
        if len(ub) == 0:
            ub = np.array([dz_ub, zt_ub])

        res = least_squares(func, x0=np.array([dz0, zt0]), jac='3-point', bounds=(lb, ub), args=(Phi_exp, kh, beta, C),
                            xtol=xtol_ls, ftol=ftol_ls, gtol=gtol_ls, max_nfev=max_nfev_ls)
        dz_opt = res.x[0]
        zt_opt = res.x[1]
        misfit = res.cost

    else:
        raise ValueError('Method undefined')

    return dz_opt, zt_opt, misfit


# def find_zb(Phi_exp, kh, beta, zt, zb0, C=0.0, wlf=False, method='fmin', lb=[], ub=[]):
#     """
#     Find depth to bottom of magnetic slab for a given radial spectrum
#
#     Parameters
#     ----------
#         Phi_exp : Spectrum values for kh
#         kh      : norm of the wave number in the horizontal plane
#         beta    : fractal parameter
#         zt      : depth of top of magnetic layer
#         zb0     : starting value
#         C       : field constant (Maus et al., 1997)
#         wlf     : apply low frequency weighting
#         method  : 'fmin' -> simplex method
#                   'ls' -> least-squares
#         lb      : lower bounds for least-squares (zb)
#         ub      : upper bounds for least-squares (zb)
#
#     Returns
#     -------
#         zb, Normalized RMS misfit
#     """
#     if not np.isscalar(wlf):
#         # wlf must be array of size Phi_exp
#         w = 1.0 / wlf
#     elif wlf:
#         w = np.linspace(1.5, 0.5, Phi_exp.size)
#     else:
#         w = 1.0
#     # define function to minimize
#     def func(zb, beta, Phi_exp, zt, kh, C):
#         dz = zb - zt
#         return np.sqrt(1.0/Phi_exp.size * np.sum((w*(Phi_exp - bouligand4(beta, zt, dz, kh, C))**2)))
#
#     if method == 'fmin':
#         xopt = fmin(func, x0=zb0, args=(beta, Phi_exp, zt, kh, C), disp=False, full_output=True, xtol=xtol_fmin, ftol=ftol_fmin, maxiter=maxiter_fmin, maxfun=maxfun_fmin)
#         zb_opt = xopt[0]
#         misfit = xopt[1]
#
#     elif method == 'ls':
#         if len(lb) == 0:
#             lb = np.array([zb_lb])
#         if len(ub) == 0:
#             ub = np.array([zb_ub])
#
#         res = least_squares(func, x0=zb0, jac='3-point', bounds=(lb,ub), args=(beta, Phi_exp, zt, kh, C), xtol=xtol_ls, ftol=ftol_ls, gtol=gtol_ls, max_nfev=max_nfev_ls)
#         zb_opt = res.x[0]
#         misfit = res.cost
#
#     return zb_opt[0], misfit

def find_dz_zt_C(Phi_exp, kh, beta, dz0, zt0, C0, wlf=False, method='fmin', lb=[], ub=[]):
    """
    Find fractal model parameters for a given radial spectrum

    Parameters
    ----------
        Phi_exp : Spectrum values for kh
        kh      : norm of the wave number in the horizontal plane
        beta    : value of beta
        dz0     : starting value of thickness of magnetic layer
        zt0     : starting value of depth of top of magnetic layer
        C0      : starting value of field constant (Maus et al., 1997)
        wlf     : apply low frequency weighting
        method  : 'fmin' -> simplex method
                  'ls' -> least-squares
        lb      : lower bounds for least-squares (dz, zt, C)
        ub      : upper bounds for least-squares (dz, zt, C)

    Returns
    -------
        dz, zt, C, Normalized RMS misfit
    """
    if not np.isscalar(wlf):
        # wlf must be array of size Phi_exp
        w = 1.0 / wlf
    elif wlf:
        w = np.linspace(1.5, 0.5, Phi_exp.size)
    else:
        w = 1.0

    # define function to minimize
    def func(x, Phi_exp, kh, beta):
        dz = x[0]
        zt = x[1]
        C = x[2]
        return np.sqrt(1.0 / Phi_exp.size * np.sum((w * (Phi_exp - bouligand4(beta, zt, dz, kh, C)) ** 2)))

    if method == 'fmin':
        xopt = fmin(func, x0=np.array([dz0, zt0, C0]), args=(Phi_exp, kh, beta), full_output=True, disp=False,
                    xtol=xtol_fmin, ftol=ftol_fmin, maxiter=maxiter_fmin, maxfun=maxfun_fmin)
        dz_opt = xopt[0][0]
        zt_opt = xopt[0][1]
        C_opt = xopt[0][2]
        misfit = xopt[1]

    elif method == 'ls':
        if len(lb) == 0:
            lb = np.array([dz_lb, zt_lb, C_lb])
        if len(ub) == 0:
            ub = np.array([dz_ub, zt_ub, C_ub])

        res = least_squares(func, x0=np.array([dz0, zt0, C0]), jac='3-point', bounds=(lb, ub), args=(Phi_exp, kh, beta),
                            xtol=xtol_ls, ftol=ftol_ls, gtol=gtol_ls, max_nfev=max_nfev_ls)
        dz_opt = res.x[0]
        zt_opt = res.x[1]
        C_opt = res.x[2]
        misfit = res.cost

    elif method == '2s':

        if len(lb) == 0:
            lb = np.array([dz_lb, zt_lb, C_lb])
        if len(ub) == 0:
            ub = np.array([dz_ub, zt_ub, C_ub])

        res = least_squares(func, x0=np.array([dz0, zt0, C0]), jac='3-point', bounds=(lb, ub), args=(Phi_exp, kh, beta),
                            xtol=xtol_ls, ftol=ftol_ls, gtol=gtol_ls, max_nfev=max_nfev_ls)
        dz_opt = res.x[0]
        zt_opt = res.x[1]
        C_opt = res.x[2]

        xopt = fmin(func, x0=np.array([dz_opt, zt_opt, C_opt]), args=(Phi_exp, kh, beta), full_output=True, disp=False,
                    xtol=xtol_fmin, ftol=ftol_fmin, maxiter=maxiter_fmin, maxfun=maxfun_fmin)
        dz_opt = xopt[0][0]
        zt_opt = xopt[0][1]
        C_opt = xopt[0][2]
        misfit = xopt[1]

    elif method == '2sb':
        xopt = fmin(func, x0=np.array([dz0, zt0, C0]), args=(Phi_exp, kh, beta), full_output=True, disp=False,
                    xtol=xtol_fmin, ftol=ftol_fmin, maxiter=maxiter_fmin, maxfun=maxfun_fmin)
        dz_opt = xopt[0][0]
        zt_opt = xopt[0][1]
        C_opt = xopt[0][2]

        if len(lb) == 0:
            lb = np.array([dz_lb, zt_lb, C_lb])
        if len(ub) == 0:
            ub = np.array([dz_ub, zt_ub, C_ub])

        if dz_opt < lb[0]:
            dz_opt = lb[0]
        if dz_opt > ub[0]:
            dz_opt = ub[0]
        if zt_opt < lb[1]:
            zt_opt = lb[1]
        if zt_opt > ub[1]:
            zt_opt = ub[1]
        if C_opt < lb[2]:
            C_opt = lb[2]
        if C_opt > ub[2]:
            C_opt = ub[2]

        res = least_squares(func, x0=np.array([dz_opt, zt_opt, C_opt]), jac='3-point', bounds=(lb, ub),
                            args=(Phi_exp, kh, beta), xtol=xtol_ls, ftol=ftol_ls, gtol=gtol_ls, max_nfev=max_nfev_ls)
        dz_opt = res.x[0]
        zt_opt = res.x[1]
        C_opt = res.x[2]
        misfit = res.cost

    else:
        raise ValueError('Method undefined')

    return dz_opt, zt_opt, C_opt, misfit


def find_zt_C(Phi_exp, kh, beta, dz, zt0, C0, wlf=False, method='fmin', lb=[], ub=[]):
    """
    Find fractal model parameters for a given radial spectrum

    Parameters
    ----------
        Phi_exp : Spectrum values for kh
        kh      : norm of the wave number in the horizontal plane
        beta    : value of beta
        dz      : thickness of magnetic layer
        zt0     : starting value of depth of top of magnetic layer
        C0      : starting value of field constant (Maus et al., 1997)
        wlf     : apply low frequency weighting
        method  : 'fmin' -> simplex method
                  'ls' -> least-squares
        lb      : lower bounds for least-squares (zt, C)
        ub      : upper bounds for least-squares (zt, C)

    Returns
    -------
        zt, C, Normalized RMS misfit
    """
    if not np.isscalar(wlf):
        # wlf must be array of size Phi_exp
        w = 1.0 / wlf
    elif wlf:
        w = np.linspace(1.5, 0.5, Phi_exp.size)

    else:
        w = 1.0

    # define function to minimize
    def func(x, Phi_exp, kh, dz, beta):
        zt = x[0]
        C = x[1]
        return np.sqrt(1.0 / Phi_exp.size * np.sum((w * (Phi_exp - bouligand4(beta, zt, dz, kh, C)) ** 2)))

    if method == 'fmin':
        xopt = fmin(func, x0=np.array([zt0, C0]), args=(Phi_exp, kh, dz, beta), full_output=True, disp=False,
                    xtol=xtol_fmin, ftol=ftol_fmin, maxiter=maxiter_fmin, maxfun=maxfun_fmin)
        zt_opt = xopt[0][0]
        C_opt = xopt[0][1]
        misfit = xopt[1]

    elif method == 'ls':
        if len(lb) == 0:
            lb = np.array([zt_lb, C_lb])
        if len(ub) == 0:
            ub = np.array([zt_ub, C_ub])

        res = least_squares(func, x0=np.array([zt0, C0]), jac='3-point', bounds=(lb, ub), args=(Phi_exp, kh, dz, beta),
                            xtol=xtol_ls, ftol=ftol_ls, gtol=gtol_ls, max_nfev=max_nfev_ls)
        zt_opt = res.x[0]
        C_opt = res.x[1]
        misfit = res.cost

    else:
        raise ValueError('Method undefined')

    return zt_opt, C_opt, misfit


def hfu(x):
    """
    Transform HFU (heat flow unit) into mW/m2
    """
    return x * 41.86


def hgu(x):
    """
    Transform HGU (heat generation unit) into µW/m3
    """
    return x * 0.418


def lachenbruch(Q0, A0, k, z, D=7500.0):
    """
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
    """
    return z * (Q0 * 1.e-3 - D * A0 * 1.e-6) / k + (D * D * A0 * 1.e-6 * (1. - np.exp(-z / D))) / k


def lachenbruch_z(Q0, A0, k, T, D=7500.0):
    """
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
    """
    Q = Q0 * 1.e-3
    A = A0 * 1.e-6
    z = (A * D ** 2 + D * (A * D - Q) * lambertw(
        -A * D * np.exp((-A * D ** 2 + k * T) / (D * (A * D - Q))) / (A * D - Q)) - k * T) / (A * D - Q)
    if np.isreal(z):
        return z.real
    else:
        raise ValueError('Complex depth returned')


def find_zb_lach(T, Q0, A0, k, z1, z2, D=7500.0):
    """
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
    """

    def func(z, T, Q0, A0, k, D):
        return np.abs(T - lachenbruch(Q0, A0, k, z, D))

    z_opt = fminbound(func, z1, z2, args=(T, Q0, A0, k, D), disp=0)
    return z_opt


def find_D_lach(T, z, Q0, A0, k, D1, D2):
    """
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
    """

    def func(D, T, Q0, A0, k, z):
        return np.abs(T - lachenbruch(Q0, A0, k, z, D))

    D_opt = fminbound(func, D1, D2, args=(T, Q0, A0, k, z), disp=0)
    return D_opt


def find_zb_okubo(S, k, k_cut):
    ind = k > k_cut
    x = k[ind]
    y = S[ind]
    A = np.vstack([x, np.ones(len(x))]).T
    zt, c = lstsq(A, y)[0]

    ind = np.logical_not(ind)

    x = k[ind]
    G = np.log(np.exp(S[ind]) / (x * x))
    y = G
    A = np.vstack([x, np.ones(len(x))]).T
    zo, c = lstsq(A, y)[0]
    zo = -zo;
    zt = -zt
    return 2 * zo - zt


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
    print('Time : ', str(t2 - t1))
    zb = zt + dz
    beta_opt, fopt = find_beta(dz, Phi_exp, kh, beta0=1.5, C=C)
    print(beta_opt, fopt)

    beta_opt, zt_opt, fopt = find_beta_zt(dz, Phi_exp, kh, beta0=1.5, zt0=0.5, C=C)
    print(beta_opt, zt_opt, fopt)

    dz_opt, fopt = find_dz(15, Phi_exp, kh, beta_opt, zt_opt, C=C)
    print('dz_opt = ' + str(dz_opt))

    # 5% noise
    Phi_exp += 0.05 * np.random.rand(Phi_exp.shape[0]) * Phi_exp
    beta_opt, fopt = find_beta(dz, Phi_exp, kh, beta0=1.5, C=C)
    print(beta_opt, fopt)
    beta_opt, zt_opt, fopt = find_beta_zt(dz, Phi_exp, kh, beta0=1.5, zt0=0.5, C=C)
    print(beta_opt, zt_opt, fopt)

    dz_opt, fopt = find_dz(15, Phi_exp, kh, beta_opt, zt_opt, C=5.0)
    print('dz_opt = ' + str(dz_opt))

    zb = find_zb_okubo(Phi_exp, kh / (2 * np.pi), 0.05)
    print(zb)

    show_plots = False

    if show_plots:
        plt.subplot(1, 3, 1)
        for v in np.arange(0.0, 2.5, 0.5):
            plt.semilogx(kh, bouligand4(beta, v, dz, kh, C), 'k')
            plt.hold(True)
        plt.xlim(0.001, 3)

        C = -9.0
        plt.subplot(1, 3, 2)
        for v in [10.0, 20.0, 50.0, 100.0, 200.0]:
            plt.semilogx(kh, bouligand4(beta, zt, v, kh, C), 'k')
            plt.hold(True)
        plt.semilogx(kh, C - 2.0 * kh * zt - (beta - 1.0) * np.log(kh), 'r')
        plt.xlim(0.001, 3)

        plt.subplot(1, 3, 3)
        for v in np.arange(0.0, 5.0):
            plt.semilogx(kh, bouligand4(v, zt, dz, kh, C), 'k')
            plt.hold(True)
        plt.semilogx(kh, C - 2.0 * kh * zt + 2.0 * np.log(1.0 - np.exp(-kh * dz)), 'r')
        plt.xlim(0.001, 3)

        plt.show()

    T = 580.0
    Q0 = 20.0
    A0 = 0.5
    k = 2.5
    z0 = 20000.0

    z = find_zb_lach(T, Q0, A0, k, 10000.0, 100000.0)
    z1 = lachenbruch_z(Q0, A0, k, T)
    print('Lach: ' + str(z) + '      ' + str(z1))

    Tz = lachenbruch(Q0, A0, k, z)
    Tz1 = lachenbruch(Q0, A0, k, z1)
    print(T - Tz, T - Tz1)

    testFFTMA = False
    testSpec = True
    testAz = False

    if testFFTMA:
        grid = Grid2d('')
        grid.ncol = 1024
        grid.nrow = 2048
        grid.dx = 0.5
        grid.dy = 0.5

        cm = [geostat.CovarianceNugget(0.2), geostat.CovarianceSpherical(np.array([250.0, 200.0]), np.array([0]), 2.5)]

        G = grid.preFFTMA(cm)

        Z = grid.FFTMA()
        plt.matshow(Z.T)
        plt.show()

    if testSpec:
        g = Grid2d(
            '+proj=lcc +lat_1=49 +lat_2=77 +lat_0=63 +lon_0=-92 +x_0=0 +y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs')
        g.read_nc('/Users/giroux/JacquesCloud/Projets/CPD/NAmag/Qc_lcc_k.nc')

        #         S, k, std, ns, flag = g.getRadialSpectrum(1606000.0, -1963000.0, 500000.0,
        #                                              tukey, detrend=1, cdecim=5, kcut=2.0)

        S, k, std, ns, flag = g.get_radial_spectrum(1606000.0, -1963000.0, 500000.0, tukey, detrend=1)

        S2, k2, std2, ns2, flag = g.get_radial_spectrum(1606000.0, -1963000.0, 500000.0, tukey, detrend=1, logspace=40,
                                                        padding=2)

        beta1, C1, misfit = find_beta_C(dz + zt, S, k, 3.0, 25.0)

        Phi_exp = bouligand4(beta1, zt, dz, k, C1)

        S3, k3, std3, ns3, flag = g.get_radial_spectrum(1606000.0, -1963000.0, 250000.0, tukey, detrend=1, order=5,
                                                        mem=1)

        plt.figure()
        plt.semilogx(k, S, '-', k2, S2, 'o', k3, S3, ':', k, Phi_exp, '*')
        plt.legend(('1', '2', '3', '4'))
        plt.show(block=False)

        plt.figure()
        l1, l2 = plt.semilogx(k, S, '-', k2, S2, 'o')
        l3 = plt.fill_between(k, S - std, S + std)
        l3.set_alpha(0.2)
        l4 = plt.fill_between(k2, S2 - std2, S2 + std2)
        l4.set_alpha(0.2)
        plt.show(block=False)

        plt.figure()
        plt.loglog(k, ns, '-', k2, ns2, 'o', k3, ns3, ':')
        plt.legend(('1', '2', '3'))
        plt.title('N_s')
        plt.show()

        print('Done')

    if testAz:
        g = Grid2d(
            '+proj=lcc +lat_1=49 +lat_2=77 +lat_0=63 +lon_0=-92 +x_0=0 +y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs')
        g.read_nc('/Users/giroux/JacquesCloud/Projets/CPD/NAmag/Qc_lcc_k.nc')

        S, k, theta, flag = g.get_azimuthal_spectrum(1606000.0, -1963000.0, 500000.0, tukey, 1)

        fig, ax = plt.subplots()
        ax.set_xscale('log')
        plt.pcolormesh(k, theta, S, axes=ax)
        plt.show()

#    g = Grid2d('+proj=lcc +lat_1=49 +lat_2=77 +lat_0=63 +lon_0=-92 +x_0=0 +y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs')
#    g.readnc('NAmag/Qc_lcc_k.nc')
#
#    plt.figure()
#    plt.imshow(g.data, clim=[-500.0, 600])
#
#    plt.show()
