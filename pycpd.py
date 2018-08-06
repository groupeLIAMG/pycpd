#!/usr/bin/env python3
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

import os
import shelve
import sys
import numpy as np
import pandas as pd
from scipy.signal import tukey, hanning

import matplotlib
from PyQt5.QtGui import QDoubleValidator, QIntValidator

# Make sure that we are using QT5
matplotlib.use('Qt5Agg')
from PyQt5.QtWidgets import (QWidget, QPushButton, QHBoxLayout, QVBoxLayout, QMainWindow, QCheckBox,
                             QApplication, QGroupBox, QLabel, QLineEdit, QComboBox, QFileDialog,
                             QGridLayout, QSlider, QSizePolicy, QAction, qApp, QFrame, QMessageBox,
                             QTabWidget, QDialog, QListWidget, QListWidgetItem)
from PyQt5.QtCore import Qt, pyqtSignal, QLocale

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from mpl_toolkits.basemap import Basemap, cm  # @UnresolvedImport

import cpd

locale = QLocale()

class MyMplCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""
    # from http://matplotlib.org/examples/user_interfaces/embedding_in_qt5.html
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)

        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)



class SpectrumCanvas(MyMplCanvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.l1 = None     # line of measured spectrum
        self.l2 = None     # line of modelled spectrum
        self.t = None
        self.qm = None     # Quad mesh returned by pcolormesh
        
    def plot(self, f, mauspar, logscale, lfw):
        if self.qm != None:
            self.axes.cla()
            self.qm = None
        (beta, zt, dz, C) = mauspar
        Sm = cpd.bouligand4(beta, zt, dz, f.k_r, C)
        el1 = np.linalg.norm(f.S_r-Sm)
        if lfw:
            w = np.linspace(2.0, 1.0, f.k_r.size)
            w /= w.sum()
            el2 = np.linalg.norm(w*(f.S_r-Sm))
        if self.l1 == None:
            if logscale:
                self.l1, self.l2 = self.axes.semilogx(f.k_r, f.S_r, f.k_r, Sm)
                self.l3 = self.axes.fill_between(f.k_r, f.S_r-f.E2, f.S_r+f.E2,
                                                 facecolor=[0.12156863, 0.46666667, 0.70588235, 0.2])
            else:
                self.l1, self.l2 = self.axes.plot(f.k_r, f.S_r, f.k_r, Sm)
                self.l3 = self.axes.fill_between(f.k_r, f.S_r-f.E2, f.S_r+f.E2,
                                                 facecolor=[0.12156863, 0.46666667, 0.70588235, 0.2])
            if lfw:
                self.t.set_text('L-2 misfit: {0:g} (lfw: {1:g})'.format(el1, el2))
            else:
                self.t = self.axes.set_title('L-2 misfit: {0:g}'.format(el1))
            self.axes.set_xlabel('Wavenumber (rad/km)')
            self.axes.set_ylabel('Radial Spectrum')
        else:
            if logscale:
                self.axes.set_xscale('log')
            else:
                self.axes.set_xscale('linear')
            self.l1.set_data(f.k_r, f.S_r)
            self.l2.set_data(f.k_r, Sm)
            self.l3.remove()
            self.l3 = self.axes.fill_between(f.k_r, f.S_r-f.E2, f.S_r+f.E2,
                                             facecolor=[0.12156863, 0.46666667, 0.70588235, 0.2])

            self.axes.relim()
            self.axes.autoscale_view()
            
            if lfw:
                self.t.set_text('L-2 misfit: {0:g} (lfw: {1:g})'.format(el1, el2))
            else:
                self.t.set_text('L-2 misfit: {0:g}'.format(el1))
                 
        self.draw()
    
    def plot2D(self, f, logscale):
        if self.l1 != None:
            self.l1 = None
            self.l2 = None
            self.axes.cla()
            
        if logscale:
            self.axes.set_xscale('log')
        else:
            self.axes.set_xscale('linear')
        self.qm = self.axes.pcolormesh(f.k_r, f.theta, f.S_r)
        self.axes.set_xlabel('Wavenumber (rad/km)')
        self.axes.set_ylabel('Angle (°)')
        self.draw()

class MapCanvas(MyMplCanvas):
    
    mapClicked = pyqtSignal(float, float)
    bhClicked = pyqtSignal(int)
    bhRightClicked = pyqtSignal(int)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bh1 = None
        self.m = None
        self.im = None
        self.cb = None
        self.allbh = None
        self.gxmin = 0.0
        self.gymin = 0.0
        self.amin = -600.0
        self.amax = 600.0
        
    def drawMap(self, grid):
        
        self.axes.cla()
        if self.cb != None:
            self.cb.remove()
        
        bmpar = cpd.proj4string2dict(grid.proj4string)
        bmpar['width'] = grid.xeast-grid.xwest
        bmpar['height'] = grid.ynorth-grid.ysouth
        bmpar['llcrnrlon'] = grid.c0[0]
        bmpar['llcrnrlat'] = grid.c0[1]
        bmpar['urcrnrlon'] = grid.c1[0]
        bmpar['urcrnrlat'] = grid.c1[1]
        bmpar['resolution'] = 'l'
        bmpar['ax'] = self.axes
        
        self.gxmin = grid.xwest
        self.gymin = grid.ysouth
        self.m = Basemap(**bmpar)
        self.m.drawcoastlines()
        self.m.drawstates()
        self.m.drawcountries()
        self.m.drawmeridians(np.arange(0,360,10),labels=[0,0,0,1],fontsize=10)
        self.m.drawparallels(np.arange(-90,90,10),labels=[1,0,0,0],fontsize=10)
        
        self.im = self.m.imshow(grid.data, cmap=cm.GMT_wysiwyg, clim=(self.amin, self.amax), picker=1)
        self.cb = self.fig.colorbar(self.im, ticks=np.arange(self.amin, self.amax, (self.amax-self.amin)/10.0))
        self.axes.set_title('Magnetic Anomaly')
        
        self.draw()
        
        def onpick(event):
            if event.artist == self.im and event.mouseevent.button == 1:
                self.mapClicked.emit(grid.xwest+event.mouseevent.xdata,
                                     grid.ysouth+event.mouseevent.ydata)
            elif event.artist == self.allbh:
                if event.artist.get_visible():
                    if event.mouseevent.button == 3:
                        self.bhRightClicked.emit(event.ind[0])
                    else:
                        self.bhClicked.emit(event.ind[0])

                    
        self.mpl_connect('pick_event', onpick)
        
    def set_clim(self, amin, amax):
        self.amin = amin
        self.amax = amax
        if self.im != None:
            self.im.set_clim((amin, amax))
            self.draw()
        
    def updateBhLoc(self, f):
        if self.m != None:
            if self.bh1 == None:
                self.bh1, = self.m.plot(f.x-self.gxmin, f.y-self.gymin, 'k*', mfc='r', ms=20)
            else:
                self.bh1.set_data(f.x-self.gxmin, f.y-self.gymin)
            self.draw()

    def updateBhLocs(self, forages, visible):
        if forages == None:
            return
        if self.m != None:
            if self.allbh == None:
                x = np.empty((len(forages),))
                y = np.empty((len(forages),))
                for n in range(len(forages)):
                    x[n] = forages[n].x-self.gxmin
                    y[n] = forages[n].y-self.gymin
                    
                self.allbh, = self.m.plot(x, y, 'ko', mfc=[0.5, 0.5, 0.5, 0.5], picker=2)
                
            self.allbh.set_visible(visible)                
            self.draw()
            
class LachenbruchCanvas(MyMplCanvas):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.l = None
        self.t = None

    def plot(self, T, z, title):
        if self.l == None:
            self.l, = self.axes.plot(T, z)
            self.t = self.axes.set_title(title)
            self.axes.invert_yaxis()
            self.axes.grid()
            self.axes.set_xlabel('T (°C)')
            self.axes.set_ylabel('Depth (km)')

        else:
            self.l.set_data(T, z)
            self.axes.relim()
            self.axes.autoscale_view()
            
            self.t.set_text(title)
            
        self.draw()

class BoreholeData(QGroupBox):
    def __init__(self):
        super().__init__('Boreholes')
        
        self.bhlist = QComboBox()
        self.down = QPushButton('\u21E6')
        self.down.setMaximumWidth(30)
        self.up = QPushButton('\u21E8')
        self.up.setMaximumWidth(30)

        self.offshore = QLabel()
        self.zb_sat = QLabel()
        self.zb_sat.setToolTip('Magnetic crustal thickness based on satellite magnetic field model')
        self.show = QCheckBox('Show on map')

        infos = QFrame()
        self.Q0 = QComboBox()
        self.A = QComboBox()
        self.k = QComboBox()
        
        il = QHBoxLayout()
        il.addWidget(QLabel('Q0'))
        il.addWidget(self.Q0)
        il.addWidget(QLabel('A'))
        il.addWidget(self.A)
        il.addWidget(QLabel('k'))
        il.addWidget(self.k)
        
        infos.setLayout(il)

        sims = QFrame()
        sims.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.showAsim = QPushButton('A')
        self.showksim = QPushButton('k')
        
        sl = QHBoxLayout()
        sl.addWidget(QLabel('Simulations'))
        sl.addWidget(self.showAsim)
        sl.addWidget(self.showksim)
        
        sims.setLayout(sl)

        gl = QGridLayout()
        
        gl.addWidget(self.bhlist,0,0)
        gl.addWidget(infos,0,1)
        gl.addWidget(self.down,0,2)
        gl.addWidget(self.up,0,3)
        gl.addWidget(self.show, 0, 4)
        gl.addWidget(self.zb_sat, 1, 0)
        gl.addWidget(sims, 1, 1)
        gl.addWidget(self.offshore, 1,4)
        self.setLayout(gl)

    def setList(self, forages):
        
        self.bhlist.clear()
        for f in forages:
            self.bhlist.addItem(f.site_name)
        
        

class SpectrumParams(QGroupBox):
    def __init__(self):
        super().__init__('Spectrum')
        
        self.type = QComboBox()
        self.type.addItems(('Radial', 'Azimuthal'))
        self.detrend = QComboBox()
        self.detrend.addItems(('None', 'Linear', 'Mean', 'Median', 'Mid'))
        self.detrend.setCurrentIndex(1)
        self.taperwin = QComboBox()
        self.taperwin.addItems(('tukey', 'hanning', 'None'))
        self.taperwin.setCurrentIndex(0)
        self.winsize = QLineEdit(locale.toString(500.0, format='f', precision=1))
        self.winsize.setValidator(QDoubleValidator())
        self.cdecim = QLineEdit('0')
        self.cdecim.setValidator(QIntValidator())
        self.kcut = QLineEdit(locale.toString(0.0, format='f', precision=1))
        self.kcut.setValidator(QDoubleValidator())
#        self.winsize.setMinimumWidth(75)
        self.log = QCheckBox('Log scale')
        self.log.setChecked(True)
        self.log.setToolTip('Display spectrum in log scale')
        self.lspace_cbox = QCheckBox('Interp. in log space')
        self.lspace_cbox.setChecked(False)
        self.lspace_edit = QLineEdit('50')
        self.lspace_edit.setValidator(QIntValidator())
        self.lspace_edit.setToolTip('Number of points for interpolation')
        
        self.estiml = QLabel('Estimator')
        self.estimator = QComboBox()
        self.estimator.addItems(('FFT','Maximum Entropy (Srinivasa)', 'Maximum Entropy (Lim-Malik)'))
#        self.estimator.addItems(('FFT','Maximum Entropy'))
        self.estimator.setMaximumWidth(200)
        self.memest =  QComboBox()
        self.memest.addItems(('FFT', 'ARMA'))
        self.order = QLineEdit('15')
        self.order.setValidator(QIntValidator())
        self.orderl = QLabel('MEM order')
        self.orderl.setAlignment(Qt.AlignRight)
        
        self.memest.setVisible(False)
        self.order.setVisible(False)
        self.orderl.setVisible(False)
        
        gl = QGridLayout()
        gl.addWidget(self.type, 0, 0, 1, 2)
        gl.addWidget(self.lspace_cbox, 0, 4, 1, 2)
        gl.addWidget(self.lspace_edit, 0, 6)
        gl.addWidget(QLabel('Size (km)'), 1, 0)
        gl.addWidget(self.winsize, 1, 1)
        gl.addWidget(QLabel('Detrending'), 1, 2)
        gl.addWidget(self.detrend, 1, 3)
        taperl = QLabel('Taper')
        taperl.setAlignment(Qt.AlignRight)
        gl.addWidget(taperl, 1, 4)
        gl.addWidget(self.taperwin, 1, 5)
        gl.addWidget(self.log, 1, 6)
        
        gl.addWidget(QLabel('Casc. decimation (npts)'), 2, 0, 1, 2)
        gl.addWidget(self.cdecim, 2, 2)
        gl.addWidget(QLabel('High-k cutoff (rad/km)'), 2, 4, 1, 2)
        gl.addWidget(self.kcut, 2, 6)

        gl.addWidget(self.estiml, 3, 0)
        gl.addWidget(self.estimator, 3, 1, 1, 3)
        gl.addWidget(self.memest, 3, 4)
        gl.addWidget(self.orderl, 3, 5)
        gl.addWidget(self.order, 3, 6)
        
        self.setLayout(gl)
        
class MausParams(QGroupBox):
    def __init__(self):
        super().__init__('Maus Model Parameters')
        
        self.bbeta = (1.0, 6.0)
        self.bzt = (0.0, 5.0)
        self.bdz = (5.0, 150.0)
        self.bC = (3.0, 30.0)
        
        self.beta_fac = int(100*(0.00001+self.bbeta[1]-self.bbeta[0]))
        self.zt_fac = int(10*(0.00001+self.bzt[1]-self.bzt[0]))
        self.dz_fac = int(10*(0.00001+self.bdz[1]-self.bdz[0]))
        self.C_fac = int(10*(0.00001+self.bC[1]-self.bC[0]))
        
        self.initUI()
        
    def initUI(self):
        gl = QGridLayout()
        
        beta = 3.0
        zt = 1.0
        dz = 20.0
        C = 25.0
        
        self.betasl = QSlider(Qt.Horizontal)
        self.betasl.setMinimumWidth(100)
        self.betasl.setMaximum(self.beta_fac)
        val = int((beta-self.bbeta[0])/(self.bbeta[1]-self.bbeta[0]) * self.beta_fac)
        self.betasl.setValue(val)
        
        self.ztsl = QSlider(Qt.Horizontal)  
        self.ztsl.setMinimumWidth(100)
        self.ztsl.setMaximum(self.zt_fac)
        val = int((zt-self.bzt[0])/(self.bzt[1]-self.bzt[0]) * self.zt_fac)
        self.ztsl.setValue(val)
        
        self.dzsl = QSlider(Qt.Horizontal)
        self.dzsl.setMinimumWidth(100)
        self.dzsl.setMaximum(self.dz_fac)
        val = int((dz-self.bdz[0])/(self.bdz[1]-self.bdz[0]) * self.dz_fac)
        self.dzsl.setValue(val)
        
        self.Csl = QSlider(Qt.Horizontal)
        self.Csl.setMinimumWidth(100)
        self.Csl.setMaximum(self.C_fac)
        val = int((C-self.bC[0])/(self.bC[1]-self.bC[0]) * self.C_fac)
        self.Csl.setValue(val)
                
        self.betaed = QLineEdit(locale.toString(beta, format='f', precision=2))
        self.betaed.setMinimumWidth(75)
        self.betaed.setValidator(QDoubleValidator())
        self.zted = QLineEdit(locale.toString(zt, format='f', precision=1))
        self.zted.setMinimumWidth(75)
        self.zted.setValidator(QDoubleValidator())
        self.dzed = QLineEdit(locale.toString(dz, format='f', precision=1))
        self.dzed.setMinimumWidth(75)
        self.dzed.setValidator(QDoubleValidator())
        self.Ced = QLineEdit(locale.toString(C, format='f', precision=1))
        self.Ced.setMinimumWidth(75)
        self.Ced.setValidator(QDoubleValidator())
        
        self.betac = QCheckBox()
        self.ztc = QCheckBox()
        self.dzc = QCheckBox()
        self.Cc = QCheckBox()
        self.lfc = QCheckBox('Low frequency weighting')
        self.lfc.setChecked(False)
        
        self.beta_lb = QLineEdit(locale.toString(1.5, format='f', precision=1))
        self.beta_lb.setEnabled(False)
        self.beta_lb.setValidator(QDoubleValidator())
        self.beta_ub = QLineEdit(locale.toString(5.8, format='f', precision=1))
        self.beta_ub.setEnabled(False)
        self.beta_ub.setValidator(QDoubleValidator())
        self.zt_lb = QLineEdit(locale.toString(0.0, format='f', precision=1))
        self.zt_lb.setEnabled(False)
        self.zt_lb.setValidator(QDoubleValidator())
        self.zt_ub = QLineEdit(locale.toString(5.0, format='f', precision=1))
        self.zt_ub.setEnabled(False)
        self.zt_ub.setValidator(QDoubleValidator())
        self.dz_lb = QLineEdit(locale.toString(3.0, format='f', precision=1))
        self.dz_lb.setEnabled(False)
        self.dz_lb.setValidator(QDoubleValidator())
        self.dz_ub = QLineEdit(locale.toString(80.0, format='f', precision=1))
        self.dz_ub.setEnabled(False)
        self.dz_ub.setValidator(QDoubleValidator())
        self.C_lb = QLineEdit(locale.toString(-10.0, format='f', precision=1))
        self.C_lb.setEnabled(False)
        self.C_lb.setValidator(QDoubleValidator())
        self.C_ub = QLineEdit(locale.toString(50.0, format='f', precision=1))
        self.C_ub.setEnabled(False)
        self.C_ub.setValidator(QDoubleValidator())
        
        optimfr = QGroupBox('Optimization method')
        
        self.optim = QComboBox()
        self.optim.addItem('Simplex', 'fmin')
        self.optim.addItem('least-squares', 'ls')
        self.optim.currentIndexChanged.connect(self.optimChanged)
        
        ol = QVBoxLayout()
        ol.addWidget(self.optim)
        
        optimfr.setLayout(ol)
        
        
        self.fit2step = QPushButton('2-step fit')
        self.fit2step.setToolTip('Find first β and C, then ∆z and zt')
        self.fit = QPushButton('Fit to spectrum')
        
        self.betasl.valueChanged.connect(self.betaslChanged)
        self.ztsl.valueChanged.connect(self.ztslChanged)
        self.dzsl.valueChanged.connect(self.dzslChanged)
        self.Csl.valueChanged.connect(self.CslChanged)
        
        self.betaed.editingFinished.connect(self.betaeChanged)
        self.zted.editingFinished.connect(self.zteChanged)
        self.dzed.editingFinished.connect(self.dzeChanged)
        self.Ced.editingFinished.connect(self.CeChanged)
        
        gl.addWidget(QLabel('Beta'), 1, 0)
        gl.addWidget(QLabel('zt (km)'), 2, 0)
        gl.addWidget(QLabel('∆z (km)'), 3, 0)
        gl.addWidget(QLabel('C'), 4, 0)
        
        gl.addWidget(self.betasl,1,1)
        gl.addWidget(self.ztsl,2,1)
        gl.addWidget(self.dzsl,3,1)
        gl.addWidget(self.Csl,4,1)
        
        gl.addWidget(self.betaed,1,2)
        gl.addWidget(self.zted,2,2)
        gl.addWidget(self.dzed,3,2)
        gl.addWidget(self.Ced,4,2)
        
        gl.addWidget(QLabel('Fixed'),0,3)
        gl.addWidget(self.betac,1,3, Qt.AlignCenter)
        gl.addWidget(self.ztc,2,3, Qt.AlignCenter)
        gl.addWidget(self.dzc,3,3, Qt.AlignCenter)
        gl.addWidget(self.Cc,4,3, Qt.AlignCenter)
        
        gl.addWidget(QLabel('Lower'),0,4)
        gl.addWidget(QLabel('Upper'),0,5)
        
        gl.addWidget(self.beta_lb,1,4)
        gl.addWidget(self.beta_ub,1,5)
        gl.addWidget(self.zt_lb,2,4)
        gl.addWidget(self.zt_ub,2,5)
        gl.addWidget(self.dz_lb,3,4)
        gl.addWidget(self.dz_ub,3,5)
        gl.addWidget(self.C_lb,4,4)
        gl.addWidget(self.C_ub,4,5)
        
        gl.addWidget(optimfr,0,6,2,1)
        gl.addWidget(self.lfc,2,6)
        gl.addWidget(self.fit2step,3,6)
        gl.addWidget(self.fit,4,6)
        
        self.setLayout(gl)
        
    def betaslChanged(self):
        val = self.betasl.value()
        val = self.bbeta[0] + (1.0*val)/self.beta_fac * (self.bbeta[1]-self.bbeta[0])
        self.betaed.setText(locale.toString(val, format='f', precision=2))

    def ztslChanged(self):
        val = self.ztsl.value()
        val = self.bzt[0] + (1.0*val)/self.zt_fac * (self.bzt[1]-self.bzt[0])
        self.zted.setText(locale.toString(val, format='f', precision=1))

    def dzslChanged(self):
        val = self.dzsl.value()
        val = self.bdz[0] + (1.0*val)/self.dz_fac * (self.bdz[1]-self.bdz[0])
        self.dzed.setText(locale.toString(val, format='f', precision=1))

    def CslChanged(self):
        val = self.Csl.value()
        val = self.bC[0] + (1.0*val)/self.C_fac * (self.bC[1]-self.bC[0])
        self.Ced.setText(locale.toString(val, format='f', precision=1))
        
    def betaeChanged(self):
        val = locale.toDouble(self.betaed.text())[0]
        val = int((val-self.bbeta[0])/(self.bbeta[1]-self.bbeta[0]) * self.beta_fac)
        self.betasl.setValue(val)

    def zteChanged(self):
        val = locale.toDouble(self.zted.text())[0]
        val = int((val-self.bzt[0])/(self.bzt[1]-self.bzt[0]) * self.zt_fac)
        self.ztsl.setValue(val)

    def dzeChanged(self):
        val = locale.toDouble(self.dzed.text())[0]
        val = int((val-self.bdz[0])/(self.bdz[1]-self.bdz[0]) * self.dz_fac)
        self.dzsl.setValue(val)

    def CeChanged(self):
        val = locale.toDouble(self.Ced.text())[0]
        val = int((val-self.bC[0])/(self.bC[1]-self.bC[0]) * self.C_fac)
        self.Csl.setValue(val)
        
    def optimChanged(self, index):
        self.beta_lb.setEnabled(index)
        self.beta_ub.setEnabled(index)
        self.zt_lb.setEnabled(index)
        self.zt_ub.setEnabled(index)
        self.dz_lb.setEnabled(index)
        self.dz_ub.setEnabled(index)
        self.C_lb.setEnabled(index)
        self.C_ub.setEnabled(index)
    
    @property
    def method(self):
        return self.optim.currentData()
        
class LachenbruchParams(QGroupBox):
    def __init__(self):
        super().__init__('Lachenbruch and Sass Model Parameters')
        
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel('Characteristic depth (km)'))
        self.D = QLineEdit(locale.toString(7.74, format='f', precision=2))
        self.D.setValidator(QDoubleValidator())
        hbox.addWidget(self.D)
        self.override = QCheckBox('Override bh data')
        hbox.addWidget(self.override)
        hbox.addWidget(QLabel('Q0'))
        self.Q0 = QLineEdit(locale.toString(40.0, format='f', precision=1))
        self.Q0.setValidator(QDoubleValidator())
        hbox.addWidget(self.Q0)
        hbox.addWidget(QLabel('A'))
        self.A = QLineEdit(locale.toString(2.0, format='f', precision=1))
        self.A.setValidator(QDoubleValidator())
        hbox.addWidget(self.A)
        hbox.addWidget(QLabel('k'))
        self.k = QLineEdit(locale.toString(4.0, format='f', precision=1))
        self.k.setValidator(QDoubleValidator())
        hbox.addWidget(self.k)
        
        self.setLayout(hbox)

class ProjSelection(QDialog):
    def __init__(self, parent=None):
        super(ProjSelection, self).__init__(parent)
        self.setWindowTitle('Grid Projection parameters')
        self.result = ''
        #================================================= ===============================
        # listbox
        #================================================= ===============================
        self.listWidget = QListWidget()
        
        item_ls = ('Québec : +proj=lcc +lat_1=49 +lat_2=77 +lat_0=63 +lon_0=-92 +x_0=0 +y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs',
                   'DNAG : +proj=tmerc +k_0=0.926 _lat_0=0.0 +lon_0=-100.0 +a=6371204.0 +b=6371204.0 +x_0=0 +y_0=0 +units=m +no_defs')
        
        for item in item_ls:
            w_item = QListWidgetItem(item)
            self.listWidget.addItem(w_item)
        self.listWidget.itemClicked.connect(self.onSingleClick)
        self.listWidget.itemActivated.connect(self.onDoubleClick)
        layout = QGridLayout()
        row = 0
        layout.addWidget(QLabel('Predefined proj4 strings'),row,0)
        row += 1
        layout.addWidget(self.listWidget,row,0,1,3) #col span=1, row span=3
        
        row += 1
        layout.addWidget(QLabel('Custom proj4 string'),row,0)
        self.proj4string = QLineEdit()
        self.proj4string.editingFinished.connect(self.projEdited)
        row += 1
        layout.addWidget(self.proj4string,row,0,1,3)
        #================================================= ===============================
        # OK, Cancel
        #================================================= ===============================
        row +=1
        self.but_ok = QPushButton('OK')
        layout.addWidget(self.but_ok ,row,1)
        self.but_ok.clicked.connect(self.onOk)
        
        self.but_cancel = QPushButton('Cancel')
        layout.addWidget(self.but_cancel ,row,2)
        self.but_cancel.clicked.connect(self.onCancel)
        
        #================================================= ===============================
        #
        #================================================= ===============================
        self.setLayout(layout)
        self.setGeometry(300, 200, 560, 250)
    
    def onSingleClick(self, item):
        val = item.text().split(':')
        if len(val) == 2:
            self.result = val[1]
        else:
            self.result = val[0]
        self.proj4string.setText(self.result)
        self.proj4string.setCursorPosition(0)

    def onDoubleClick(self, item):
        val = item.text().split(':')
        if len(val) == 2:
            self.result = val[1]
        else:
            self.result = val[0]
        self.accept()
        return self.result
    
    def projEdited(self):
        self.result = self.proj4string.text()
    
    def onOk(self):
        self.accept()
        return self.result
    
    def onCancel(self):
        self.reject()
    
    def getValue(self):
        return self.result




class PyCPD(QMainWindow):
    
    def __init__(self):
        super().__init__()
        
        self.forage = None
        self.forages = None
        self.grid = None
    
        self.mag_path = os.path.expanduser('~/')
        self.db_path = os.path.expanduser('~/')
        
        self.initUI()
        
        
    def initUI(self):
        
        createDbAction = QAction('Create Borehole Database ...', self)
        createDbAction.setShortcut('Ctrl+D')
        createDbAction.setStatusTip('Create borehole data')
        createDbAction.triggered.connect(self.createDb)

        openDbAction = QAction('Open Borehole Database ...', self)
        openDbAction.setShortcut('Ctrl+O')
        openDbAction.setStatusTip('Load borehole data')
        openDbAction.triggered.connect(self.loadDb)

        saveDbAction = QAction('Save Borehole Database ...', self)
        saveDbAction.setShortcut('Ctrl+Shift+S')
        saveDbAction.setStatusTip('Save borehole data')
        saveDbAction.triggered.connect(self.saveDb)

        loadMapAction = QAction('Load Mag Data ...', self)
        loadMapAction.setShortcut('Ctrl+M')
        loadMapAction.setStatusTip('Load Mag data')
        loadMapAction.triggered.connect(self.loadMag)
        
        loadProjAction = QAction('Load Project ...', self)
        loadProjAction.setShortcut('Ctrl+L')
        loadProjAction.triggered.connect(self.loadProject)

        saveProjAction = QAction('Save Project ...', self)
        saveProjAction.setShortcut('Ctrl+S')
        saveProjAction.triggered.connect(self.saveProject)

        exitAction = QAction('Exit', self)        
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(qApp.quit)
        
        self.statusBar()

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(loadMapAction)
        fileMenu.addAction(createDbAction)
        fileMenu.addAction(openDbAction)
        fileMenu.addAction(saveDbAction)
        fileMenu.addAction(loadProjAction)
        fileMenu.addAction(saveProjAction)
        fileMenu.addAction(exitAction)
        
        mw = QFrame()
        self.setCentralWidget(mw)
        
        lw = QWidget()
        rw = QWidget()
        
        self.bh = BoreholeData()
        self.locmap = MapCanvas(self, width=5, height=4, dpi=100)
        self.locmap.setMinimumSize(700, 600)
        toolbar = NavigationToolbar(self.locmap, self)
        
        mapctrl = QFrame()
        mcl = QHBoxLayout()
        mcl.addStretch()
        aminl = QLabel('A min')
        aminl.setAlignment(Qt.AlignRight)
        mcl.addWidget(aminl)
        self.amin = QLineEdit(locale.toString(self.locmap.amin))
        self.amin.setValidator(QDoubleValidator())
        self.amin.setMaximumWidth(100)
        mcl.addWidget(self.amin)
        amaxl = QLabel('A max')
        amaxl.setAlignment(Qt.AlignRight)        
        mcl.addWidget(amaxl)
        self.amax = QLineEdit(locale.toString(self.locmap.amax))
        self.amax.setValidator(QDoubleValidator())
        self.amax.setMaximumWidth(100)
        mcl.addWidget(self.amax)
        mcl.addStretch()
        self.amin.editingFinished.connect(self.achanged)
        self.amax.editingFinished.connect(self.achanged)
        
        mapctrl.setLayout(mcl)
        
        self.splot = SpectrumCanvas(self, width=5, height=4, dpi=100)
        self.splot.setMinimumSize(500,450)
        self.lachplot = LachenbruchCanvas(self, width=5, height=4, dpi=100)
        self.lachplot.setMinimumSize(500,450)
        
        self.plots = QTabWidget()
        self.plots.addTab(self.splot, 'Spectrum')
        self.plots.addTab(self.lachplot, 'T vs z')
        
        
        self.sp = SpectrumParams()
        self.mp = MausParams()
        self.lach = LachenbruchParams()


        self.bh.bhlist.currentIndexChanged.connect(self.bhChanged)
        self.bh.down.clicked.connect(self.bhDown)
        self.bh.up.clicked.connect(self.bhUp)
        self.bh.show.stateChanged.connect(self.showBh)
        self.bh.showAsim.clicked.connect(self.showAsim)
        self.bh.showksim.clicked.connect(self.showksim)
        self.bh.Q0.currentIndexChanged.connect(self.plotLachenbruch)
        self.bh.A.currentIndexChanged.connect(self.plotLachenbruch)
        self.bh.k.currentIndexChanged.connect(self.plotLachenbruch)
        
        self.locmap.mapClicked.connect(self.computeSpectrum)
        self.locmap.bhRightClicked.connect(self.bhChangedOnMap)
        self.locmap.bhClicked.connect(self.bhStatus)
        
        self.mp.betasl.valueChanged.connect(self.updateSpectrum)
        self.mp.ztsl.valueChanged.connect(self.updateSpectrum)
        self.mp.dzsl.valueChanged.connect(self.updateSpectrum)
        self.mp.Csl.valueChanged.connect(self.updateSpectrum)
        self.mp.betaed.editingFinished.connect(self.updateSpectrum)
        self.mp.zted.editingFinished.connect(self.updateSpectrum)
        self.mp.dzed.editingFinished.connect(self.updateSpectrum)
        self.mp.Ced.editingFinished.connect(self.updateSpectrum)
        self.mp.fit2step.clicked.connect(self.fitSpec2step)
        self.mp.fit.clicked.connect(self.fitSpectrum)
        self.mp.lfc.stateChanged.connect(self.updateSpectrum)

        self.sp.type.currentIndexChanged.connect(self.computeSpectrum)
        self.sp.detrend.currentIndexChanged.connect(self.computeSpectrum)
        self.sp.log.stateChanged.connect(self.updateSpectrum)
        self.sp.taperwin.currentIndexChanged.connect(self.computeSpectrum)
        self.sp.winsize.editingFinished.connect(self.computeSpectrum)
        self.sp.estimator.currentIndexChanged.connect(self.computeSpectrum)
        self.sp.memest.currentIndexChanged.connect(self.computeSpectrum)
        self.sp.order.editingFinished.connect(self.computeSpectrum)
        self.sp.cdecim.editingFinished.connect(self.computeSpectrum)
        self.sp.kcut.editingFinished.connect(self.computeSpectrum)
        self.sp.lspace_cbox.stateChanged.connect(self.computeSpectrum)
        self.sp.lspace_edit.editingFinished.connect(self.computeSpectrum)
        
        self.lach.D.editingFinished.connect(self.plotLachenbruch)
        self.lach.override.stateChanged.connect(self.plotLachenbruch)
        self.lach.Q0.editingFinished.connect(self.lachEdited)
        self.lach.A.editingFinished.connect(self.lachEdited)
        self.lach.k.editingFinished.connect(self.lachEdited)


        hbox = QHBoxLayout()
        #hbox.addStretch(1)
        hbox.addWidget(lw)
        hbox.addWidget(rw)

        rvbox1 = QVBoxLayout()
        rvbox1.addWidget(self.bh)
        rvbox1.addWidget(toolbar)
        rvbox1.addWidget(self.locmap)
        rvbox1.addWidget(mapctrl)

        rvbox2 = QVBoxLayout()
        #vbox.addStretch(1)
        rvbox2.addWidget(self.plots)
        rvbox2.addWidget(self.sp)
        rvbox2.addWidget(self.mp)
        rvbox2.addWidget(self.lach)
        
        lw.setLayout(rvbox1)
        rw.setLayout(rvbox2)
        
        mw.setLayout(hbox)    
        
        self.setGeometry(100, 100, 300, 150)
        self.setWindowTitle('Curie-Point Depth Calculator')
        self.show()
        
    def achanged(self):
        if self.locmap == None:
            return
        amin = locale.toDouble(self.amin.text())[0]
        amax = locale.toDouble(self.amax.text())[0]
        if amin > amax:
            QMessageBox.warning(self, 'Warning', 'min value larger than max value, figure not updated', QMessageBox.Ok)
            self.amin.setText((locale.toString(amax-1)))
            return
        self.locmap.set_clim(amin, amax)
        
        
    def bhDown(self):
        if self.forages == None:
            return
        if self.bh.bhlist.currentIndex() > 0:
            self.bh.bhlist.setCurrentIndex(self.bh.bhlist.currentIndex()-1)
        
    def bhUp(self):
        if self.forages == None:
            return
        if self.bh.bhlist.currentIndex() < len(self.forages)-1:
            self.bh.bhlist.setCurrentIndex(1+self.bh.bhlist.currentIndex())
        
    def bhChangedOnMap(self, index):
        self.forage = None
        self.bh.bhlist.setCurrentIndex(index)
        
    def bhStatus(self, index):
        self.statusBar().clearMessage()
        self.statusBar().showMessage('Borehole: '+self.forages[index].site_name)
        
    def bhChanged(self):
        self.forage = None
        self.computeSpectrum()
        self.plotLachenbruch()
        if self.forages != None:
            f = self.forages[self.bh.bhlist.currentIndex()]

            self.bh.Q0.clear()
            for q in f.Q0:
                self.bh.Q0.addItem(locale.toString(q))
            
            self.bh.A.clear()
            if len(f.A)>0:
                for a in f.A:
                    self.bh.A.addItem(locale.toString(a))
            
            self.bh.k.clear()
            if len(f.k)>0:
                for k in f.k:
                    self.bh.k.addItem(locale.toString(k))
                    
            if f.offshore:
                self.bh.offshore.setText('Offshore, depth = {0:g} m'.format(f.bathy))
            else:
                self.bh.offshore.setText('On land')
                
            if f.zb_sat != None:
                self.bh.zb_sat.setText('Mag. crustal thick. : {0:5.1f} km'.format(f.zb_sat))
            
            self.locmap.updateBhLoc(f)
                        
    def showBh(self):
        self.locmap.updateBhLocs(self.forages, self.bh.show.isChecked())
    
    def updateSpectrum(self):
        f = None
        if self.forage != None:
            f = self.forage
        elif self.forages != None:
            f = self.forages[self.bh.bhlist.currentIndex()]
            
        if f != None:
            if self.sp.type.currentIndex() == 0:
                beta = locale.toDouble(self.mp.betaed.text())[0]
                zt = locale.toDouble(self.mp.zted.text())[0]
                dz = locale.toDouble(self.mp.dzed.text())[0]
                C = locale.toDouble(self.mp.Ced.text())[0]
                self.splot.plot(f, (beta, zt, dz, C), self.sp.log.isChecked(), self.mp.lfc.isChecked())
            else:
                self.splot.plot2D(f, self.sp.log.isChecked())
            
    def computeSpectrum(self, x=None, y=None):
        if self.grid != None:
            if self.sp.estimator.currentIndex() == 1 or self.sp.type.currentIndex() == 1:
                self.sp.memest.setVisible(True)
                if self.sp.memest.currentIndex() == 1:
                    self.sp.order.setVisible(True)
                    self.sp.orderl.setVisible(True)
                else:
                    self.sp.order.setVisible(False)
                    self.sp.orderl.setVisible(False)
            else:
                self.sp.memest.setVisible(False)
                self.sp.order.setVisible(False)
                self.sp.orderl.setVisible(False)
                
            if self.sp.type.currentIndex() == 1:
                self.sp.estiml.setVisible(False)
                self.sp.estimator.setVisible(False)
            else:
                self.sp.estiml.setVisible(True)
                self.sp.estimator.setVisible(True)

                
            f = None
            if x != None and y != None:
                self.forage = cpd.Forage()
                f = self.forage
                f.x = x
                f.y = y
                
                self.statusBar().clearMessage()
                self.statusBar().showMessage('Point at '+locale.toString(f.x)+', '+locale.toString(f.y))
                
            elif self.forage != None:
                f = self.forage
            elif self.forages != None:
                f = self.forages[self.bh.bhlist.currentIndex()]

                self.statusBar().clearMessage()
                self.statusBar().showMessage('Borehole: '+f.site_name)
                
            if f != None:
                ww = 1000.0 * locale.toDouble(self.sp.winsize.text())[0]
                if self.sp.taperwin.currentIndex() == 0:
                    win = tukey
                elif self.sp.taperwin.currentIndex() == 1:
                    win = hanning
                else:
                    win = None
                detrend = self.sp.detrend.currentIndex()
                mem = self.sp.estimator.currentIndex()
                memest = self.sp.memest.currentIndex()
                order = int(self.sp.order.text())
                cdecim = int(self.sp.cdecim.text())
                kcut = locale.toDouble(self.sp.kcut.text())[0]
                if self.sp.lspace_cbox.isChecked():
                    logspace = int(self.sp.lspace_edit.text())
                else:
                    logspace = 0
            
                if self.sp.type.currentIndex() == 0:
                    f.S_r, f.k_r, f.E2, _ = self.grid.getRadialSpectrum(f.x, f.y, ww, win, detrend,
                                                                         mem=mem, memest=memest,
                                                                         order=order, kcut=kcut,
                                                                         cdecim=cdecim, logspace=logspace)
                else:
                    f.S_r, f.k_r, f.theta, _ = self.grid.getAzimuthalSpectrum(f.x, f.y, ww, win,
                                                                              detrend, memest=memest,
                                                                              order=order)
                    
                
                self.updateSpectrum()
        
        
    def createDb(self):
        if self.grid == None:
            QMessageBox.warning(self, 'Warning', 'Magnetic data should be loaded first', QMessageBox.Ok)
            return
        fname = QFileDialog.getOpenFileName(self, 'Open file', self.db_path, 'Heat Flow Data (*.csv)')
        if fname[0]:
            forages = []
            try:
                df = pd.read_csv(fname[0], encoding = "ISO-8859-1")            
                for n in range(df.shape[0]):
                    f = cpd.Forage(float(df['Latitude'][n]), float(df['Longitude'][n]), self.grid.proj4string)
                    buffer = 250000.0  # 250 km buffer -> this to account for window used dto compute spectrum 
                                        # (should allow user to change that, either in prefs or with dialog)
                    if self.grid.inside( f, buffer ):
                        f.site_name = str(df['Site Name'][n])
                        if f in forages:
                            ind = forages.index(f)
                            f = forages[ind]
                
                        if not np.isnan( float(df['Heat Flow'][n]) ):
                            f.Q0.append( float(df['Heat Flow'][n]) )
                        if not np.isnan( float(df['Conductivity'][n]) ):
                            f.k.append( float(df['Conductivity'][n]) )
                        if not np.isnan( float(df['Heat Prod.'][n]) ):
                            f.A.append( float(df['Heat Prod.'][n]) )
                        if f not in forages:
                            forages.append(f)
                        
            except Exception as e:
                QMessageBox.warning(self, 'Warning', str(e), QMessageBox.Ok)
                return
            
            self.db_path = os.path.dirname(fname[0])
            self.forages = forages
            self.bh.setList(self.forages)
            if len(forages) > 0:
                ex.locmap.updateBhLoc(ex.forages[0])
            self.statusBar().clearMessage()
            self.statusBar().showMessage('Database created, contains '+str(len(forages))+' boreholes')
        
        
    def saveDb(self):
        if self.forages == None:
            QMessageBox.warning(self, 'Warning', 'Current database empty, nothing to save', QMessageBox.Ok)
            return
        fname = QFileDialog.getSaveFileName(self, 'Save borehole database', self.db_path, 'Database file (*.db)')
        if fname[0]:
            if fname[0][-3:] == '.db':
                filename = fname[0][:-3] # remove extension, it is added by shelve
            else:
                filename = fname[0]
            db = shelve.open(filename, 'n')
            db['forages'] = self.forages
            db.close()
            self.db_path = os.path.dirname(fname[0])
    
    def loadDb(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', self.db_path, 'Mac OS (*.db);;Linux (*.dat)')
        if fname[0]:
            try:
                if fname[1] == 'Mac OS (*.db)':
                    tmp = fname[0][:-3]
                elif fname[1] == 'Linux (*.dat)':
                    tmp = fname[0][:-4]
                db = shelve.open(tmp,'r')
                self.forages = db['forages']
                db.close()
                self.bh.setList(self.forages)
            except IOError:
                QMessageBox.warning(self, 'Warning', 'Error loading database', QMessageBox.Ok)
        
            self.db_path = os.path.dirname(fname[0])
        
    def loadMag(self):
        oldgrid = self.grid  # store current grid in case we cannot load new stuff
        
        pselect = ProjSelection()
        if pselect.exec_():
            proj4string = pselect.getValue()
        else:
            return
        
        fname = QFileDialog.getOpenFileName(self, 'Open file', self.mag_path, 'netcdf (*.nc *.grd);;USGS (*.SGD)')
        if fname[0]:
            try:
                if fname[1] == 'USGS (*.SGD)':
                    self.grig = cpd.Grid2d(proj4string)
                    self.grid.readUSGSfile(fname[0])
                elif fname[1] == 'netcdf (*.nc *.grd)':
                    self.grid = cpd.Grid2d(proj4string)
                    self.grid.readnc(fname[0])
                    
                self.mag_path = os.path.dirname(fname[0])
                self.locmap.drawMap(self.grid)
            except RuntimeError as e:
                QMessageBox.warning(self, 'Warning', str(e.args[0], 'utf-8'), QMessageBox.Ok)
                self.grid = oldgrid
            except IOError:
                QMessageBox.warning(self, 'Warning', 'Error loading mag data', QMessageBox.Ok)
                self.grid = oldgrid
                
    def loadProject(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', os.path.expanduser('~/'), 'Mac OS (*.db);;Linux (*.dat)')
        if fname[0]:
            try:
                if fname[1] == 'Mac OS (*.db)':
                    tmp = fname[0][:-3]
                elif fname[1] == 'Linux (*.dat)':
                    tmp = fname[0][:-4]
                db = shelve.open(tmp,'r')
                self.forages = db['forages']
                self.grid = db['grid']
                db.close()
                self.bh.setList(self.forages)
                ex.locmap.drawMap(ex.grid)
            except IOError:
                QMessageBox.warning(self, 'Warning', 'Error loading database', QMessageBox.Ok)
        
    
    def saveProject(self):
        if self.forages == None and self.grid == None:
            QMessageBox.warning(self, 'Warning', 'Current project empty, nothing to save', QMessageBox.Ok)
            return
        fname = QFileDialog.getSaveFileName(self, 'Save pycpd project', os.path.expanduser('~/'), 'Database file (*.db)')
        if fname[0]:
            if fname[0][-3:] == '.db':
                filename = fname[0][:-3] # remove extension, it is added by shelve
            else:
                filename = fname[0]
            db = shelve.open(filename, 'n')
            db['forages'] = self.forages
            db['grid'] = self.grid
            db.close()
                
    def fitSpectrum(self):
        
        f = None
        if self.forage != None:
            f = self.forage
        elif self.forages != None:
            f = self.forages[self.bh.bhlist.currentIndex()]
        if f != None:
            beta = locale.toDouble(self.mp.betaed.text())[0]
            zt = locale.toDouble(self.mp.zted.text())[0]
            dz = locale.toDouble(self.mp.dzed.text())[0]
            C = locale.toDouble(self.mp.Ced.text())[0]
            
            betac = self.mp.betac.isChecked()
            ztc = self.mp.ztc.isChecked()
            dzc = self.mp.dzc.isChecked()
            Cc = self.mp.Cc.isChecked()
            lfc = self.mp.lfc.isChecked()
            
            meth = self.mp.method
            
            # none fixed
            if not betac and not ztc and not dzc and not Cc:
                lb = (locale.toDouble(self.mp.beta_lb.text())[0],
                      locale.toDouble(self.mp.zt_lb.text())[0],
                      locale.toDouble(self.mp.dz_lb.text())[0],
                      locale.toDouble(self.mp.C_lb.text())[0])
                ub = (locale.toDouble(self.mp.beta_ub.text())[0],
                      locale.toDouble(self.mp.zt_ub.text())[0],
                      locale.toDouble(self.mp.dz_ub.text())[0],
                      locale.toDouble(self.mp.C_ub.text())[0])
                x = cpd.find_beta_zt_dz_C(f.S_r, f.k_r, beta, zt, dz, C, lfc, meth, lb, ub)
                beta = x[0]
                zt = x[1]
                dz = x[2]
                C = x[3]
            # beta alone not fixed
            elif not betac and ztc and dzc and Cc:
                lb = (locale.toDouble(self.mp.beta_lb.text())[0])
                ub = (locale.toDouble(self.mp.beta_ub.text())[0])
                x = cpd.find_beta(dz, f.S_r, f.k_r, beta, zt, C, lfc, meth, lb, ub)
                beta = x[0]
            # zt alone not fixed
            elif betac and not ztc and dzc and Cc:
                lb = (locale.toDouble(self.mp.zt_lb.text())[0])
                ub = (locale.toDouble(self.mp.zt_ub.text())[0])
                x = cpd.find_zt(dz, f.S_r, f.k_r, beta, zt, C, lfc, meth, lb, ub)
                zt = x[0]
            # dz alone not fixed
            elif betac and ztc and not dzc and Cc:
                lb = (locale.toDouble(self.mp.dz_lb.text())[0])
                ub = (locale.toDouble(self.mp.dz_ub.text())[0])
                x = cpd.find_dz(dz, f.S_r, f.k_r, beta, zt, C, lfc, meth, lb, ub)
                dz = x[0]
            # C alone not fixed
            elif betac and ztc and dzc and not Cc:
                lb = (locale.toDouble(self.mp.C_lb.text())[0])
                ub = (locale.toDouble(self.mp.C_ub.text())[0])
                x = cpd.find_C(dz, f.S_r, f.k_r, beta, zt, C, lfc, meth, lb, ub)
                C = x[0]
            # beta alone fixed
            elif betac and not ztc and not dzc and not Cc:
                lb = (locale.toDouble(self.mp.dz_lb.text())[0],
                      locale.toDouble(self.mp.zt_lb.text())[0],
                      locale.toDouble(self.mp.C_lb.text())[0])
                ub = (locale.toDouble(self.mp.dz_ub.text())[0],
                      locale.toDouble(self.mp.zt_ub.text())[0],
                      locale.toDouble(self.mp.C_ub.text())[0])
                x = cpd.find_dz_zt_C(f.S_r, f.k_r, beta, dz, zt, C, lfc, meth, lb, ub)
                dz = x[0]
                zt = x[1]
                C = x[2]
            # dz alone fixed
            elif not betac and not ztc and dzc and not Cc:
                lb = (locale.toDouble(self.mp.beta_lb.text())[0],
                      locale.toDouble(self.mp.zt_lb.text())[0],
                      locale.toDouble(self.mp.C_lb.text())[0])
                ub = (locale.toDouble(self.mp.beta_ub.text())[0],
                      locale.toDouble(self.mp.zt_ub.text())[0],
                      locale.toDouble(self.mp.C_ub.text())[0])
                x = cpd.find_beta_zt_C(f.S_r, f.k_r, beta, zt, C, dz, lfc, meth, lb, ub)
                beta = x[0]
                zt = x[1]
                C = x[2]
            # C alone fixed
            elif not betac and not ztc and not dzc and Cc:
                lb = (locale.toDouble(self.mp.beta_lb.text())[0],
                      locale.toDouble(self.mp.dz_lb.text())[0],
                      locale.toDouble(self.mp.zt_lb.text())[0])
                ub = (locale.toDouble(self.mp.beta_ub.text())[0],
                      locale.toDouble(self.mp.dz_ub.text())[0],
                      locale.toDouble(self.mp.zt_ub.text())[0])
                x = cpd.find_beta_dz_zt(f.S_r, f.k_r, beta, dz, zt, C, lfc, meth, lb, ub)
                beta = x[0]
                dz = x[1]
                zt = x[2]
            # zt and dz fixed
            elif not betac and ztc and dzc and not Cc:
                lb = (locale.toDouble(self.mp.beta_lb.text())[0],
                      locale.toDouble(self.mp.C_lb.text())[0])
                ub = (locale.toDouble(self.mp.beta_ub.text())[0],
                      locale.toDouble(self.mp.C_ub.text())[0])
                x = cpd.find_beta_C(dz, f.S_r, f.k_r, beta, C, zt, lfc, meth, lb, ub)
                beta = x[0]
                C = x[1]
            # C and dz fixed
            elif not betac and not ztc and dzc and Cc:
                lb = (locale.toDouble(self.mp.beta_lb.text())[0],
                      locale.toDouble(self.mp.zt_lb.text())[0])
                ub = (locale.toDouble(self.mp.beta_ub.text())[0],
                      locale.toDouble(self.mp.zt_ub.text())[0])
                x = cpd.find_beta_zt(dz, f.S_r, f.k_r, beta, zt, C, lfc, meth, lb, ub)
                beta = x[0]
                zt = x[1]
            # beta and C fixed
            elif betac and not ztc and not dzc and Cc:
                lb = (locale.toDouble(self.mp.dz_lb.text())[0],
                      locale.toDouble(self.mp.zt_lb.text())[0])
                ub = (locale.toDouble(self.mp.dz_ub.text())[0],
                      locale.toDouble(self.mp.zt_ub.text())[0])
                x = cpd.find_dz_zt(f.S_r, f.k_r, dz, zt, beta, C, lfc, meth, lb, ub)
                dz = x[0]
                zt = x[1]
            # beta and dz fixed
            elif betac and not ztc and dzc and not Cc:
                lb = (locale.toDouble(self.mp.zt_lb.text())[0],
                      locale.toDouble(self.mp.C_lb.text())[0])
                ub = (locale.toDouble(self.mp.zt_ub.text())[0],
                      locale.toDouble(self.mp.C_ub.text())[0])
                x = cpd.find_zt_C(f.S_r, f.k_r, beta, dz, zt, C, lfc, meth, lb, ub)
                zt = x[0]
                C = x[1]
            else:
                QMessageBox.warning(self, 'Warning', 'Combination of fixed parameters not implemented', QMessageBox.Ok)
                return
                
            
            self.mp.betaed.setText(locale.toString(beta, format='f', precision=2))
            self.mp.betaeChanged()
            self.mp.zted.setText(locale.toString(zt, format='f', precision=1))
            self.mp.zteChanged()
            self.mp.dzed.setText(locale.toString(dz, format='f', precision=1))
            self.mp.dzeChanged()
            self.mp.Ced.setText(locale.toString(C, format='f', precision=1))
            self.mp.CeChanged()
            self.updateSpectrum()
    
    def fitSpec2step(self):
        f = None
        if self.forage != None:
            f = self.forage
        elif self.forages != None:
            f = self.forages[self.bh.bhlist.currentIndex()]
        if f != None:
            beta = locale.toDouble(self.mp.betaed.text())[0]
            zt = locale.toDouble(self.mp.zted.text())[0]
            dz = locale.toDouble(self.mp.dzed.text())[0]
            C = locale.toDouble(self.mp.Ced.text())[0]
            lfc = self.mp.lfc.isChecked()
            
            meth = self.mp.method

            lb = (locale.toDouble(self.mp.beta_lb.text())[0],
                  locale.toDouble(self.mp.C_lb.text())[0])
            ub = (locale.toDouble(self.mp.beta_ub.text())[0],
                  locale.toDouble(self.mp.C_ub.text())[0])
            x = cpd.find_beta_C(dz, f.S_r, f.k_r, beta, C, zt, lfc, meth, lb, ub)
            beta = x[0]
            C = x[1]

            lb = (locale.toDouble(self.mp.dz_lb.text())[0],
                  locale.toDouble(self.mp.zt_lb.text())[0])
            ub = (locale.toDouble(self.mp.dz_ub.text())[0],
                  locale.toDouble(self.mp.zt_ub.text())[0])
            x = cpd.find_dz_zt(f.S_r, f.k_r, dz, zt, beta, C, lfc, meth, lb, ub)
            dz = x[0]
            zt = x[1]
            
            self.mp.betaed.setText(locale.toString(beta, format='f', precision=2))
            self.mp.betaeChanged()
            self.mp.zted.setText(locale.toString(zt, format='f', precision=1))
            self.mp.zteChanged()
            self.mp.dzed.setText(locale.toString(dz, format='f', precision=1))
            self.mp.dzeChanged()
            self.mp.Ced.setText(locale.toString(C, format='f', precision=1))
            self.mp.CeChanged()
            self.updateSpectrum()
            
    def showAsim(self):
        if self.forages != None:
            f = self.forages[self.bh.bhlist.currentIndex()]
            if len(f.A_sim)>0:
                fig = plt.figure()
                fig.set_size_inches(7,5)
                plt.hist(f.A_sim, bins=30, label='Median: '+locale.toString(np.median(f.A_sim)))
                plt.title(f.site_name+' - Heat production')
                plt.legend()
                plt.show()
            else:
                QMessageBox.warning(self, 'Warning', 'No heat production simulation performed at this site', QMessageBox.Ok)
                
    def showksim(self):
        if self.forages != None:
            f = self.forages[self.bh.bhlist.currentIndex()]
            if len(f.k_sim)>0:
                fig = plt.figure()
                fig.set_size_inches(7, 5)
                plt.hist(f.k_sim, bins=30, label='Median: '+locale.toString(np.median(f.k_sim)))
                plt.title(f.site_name+' - Thermal conductivity')
                plt.legend()
                plt.show()
            else:
                QMessageBox.warning(self, 'Warning', 'No thermal conductivity simulation performed at this site',
                                    QMessageBox.Ok)
                
    def plotLachenbruch(self):
        if self.forages != None:
            f = self.forages[self.bh.bhlist.currentIndex()]
            z = 1000.0 * np.arange(81)            
            D = 1000.0 * locale.toDouble(self.lach.D.text())[0]
            if self.bh.Q0.count() == 0:
                return  # bh not yet displayed
        
            if self.lach.override.isChecked():
                Q0 = locale.toDouble(self.lach.Q0.text())[0]
                A = locale.toDouble(self.lach.A.text())[0]
                k = locale.toDouble(self.lach.k.text())[0]
            else:
                if f.Q0.size == 1:
                    Q0 = f.Q0[0]
                else:
                    Q0 = f.Q0[self.bh.Q0.currentIndex()]
                    
                if f.A.size == 1:
                    A = f.A[0]
                elif f.A.size > 1:
                    A = f.A[self.bh.A.currentIndex()]
                else:
                    A = np.median(f.A_sim)
                
                if f.k.size == 1:
                    k = f.k[0]
                elif f.k.size > 1:
                    k = f.k[self.bh.k.currentIndex()]
                else:
                    k = np.median(f.k_sim)
            
            T = cpd.lachenbruch(Q0, A, k, z, D)
            if T[-1] < 600.0:
                z *= 2
            T = cpd.lachenbruch(Q0, A, k, z, D)
            ind = T<=600.0
            
            self.lachplot.plot(T[ind], 0.001*z[ind], title='$A_0$={0:4.2f}, $k$={1:4.2f}'.format(A, k))
            
            
    def lachEdited(self):
        if self.lach.override.isChecked():
            self.plotLachenbruch()
                
if __name__ == '__main__':
    
    app = QApplication(sys.argv)
    ex = PyCPD()
    
    if os.path.isfile('/Users/giroux/JacquesCloud/Projets/CPD/databases/forages.db'):
          
        db = shelve.open('/Users/giroux/JacquesCloud/Projets/CPD/databases/forages','r')
        ex.forages = db['forages']
        db.close()
        ex.bh.setList(ex.forages)
           
        ex.grid = cpd.Grid2d('+proj=lcc +lat_1=49 +lat_2=77 +lat_0=63 +lon_0=-92 +x_0=0 +y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs')
        ex.grid.readnc('/Users/giroux/JacquesCloud/Projets/CPD/NAmag/Qc_lcc_k_cut.nc')
        ex.locmap.drawMap(ex.grid)
        ex.locmap.updateBhLoc(ex.forages[0])
        ex.computeSpectrum()
        ex.plotLachenbruch()
    
    sys.exit(app.exec_())
