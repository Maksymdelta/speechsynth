import matplotlib as mpl
import numpy as np
import scipy.signal as sp
import pylab as pl
import voicedb
import synther
from common import *
import traceback
import signal
import sys
import matplotlib as mpl
import stft
import lpc
import correcter
import pyin, hnm, envelope

mpl.pyplot.rcParams['keymap.fullscreen'] = ''
mpl.pyplot.rcParams['keymap.save'] = ''
mpl.pyplot.rcParams['keymap.home'] = ''
mpl.pyplot.rcParams['keymap.back'] = ''
mpl.pyplot.rcParams['keymap.forward'] = ''
mpl.pyplot.rcParams['keymap.pan'] = ''
mpl.pyplot.rcParams['keymap.zoom'] = ''
mpl.pyplot.rcParams['keymap.quit'] = ''
mpl.pyplot.rcParams['keymap.grid'] = ''
mpl.pyplot.rcParams['keymap.xscale'] = ''
mpl.pyplot.rcParams['keymap.yscale'] = ''
mpl.pyplot.rcParams['keymap.all_axes'] = ''

def wavTrim(x, threshold = -50.0):
    absX = np.abs(x)
    normFac = np.max(absX)
    if(normFac == 0.0):
        return np.zeros(0, dtype = np.float64)
    x = x / np.max(np.abs(x))
    need = absX > np.power(10, threshold / 20.0)
    idx = np.arange(len(x))[need]
    return x[idx[0]:idx[-1]]

def cutSilent(x, sinusoidEnergyList, noiseEnergyList, hopSize, sr):
    nX = len(x)
    nHop = len(sinusoidEnergyList)
    energyList = sinusoidEnergyList + noiseEnergyList
    normFac = np.max(np.abs(energyList))
    if(normFac == 0.0):
        pass
    else:
        energyList /= normFac
    need = energyList > 0.0
    energyList[need] = np.log10(energyList[need]) * 20.0
    energyList[~need] = -np.inf
    ok = np.arange(nHop)[energyList > -40.0]
    return ok[0], ok[-1]

def addPhone(path, filename, type, prev, curr, next, method = 'qfft'):
    sr, w = loadWav(filename)
    if(len(w.shape) == 2):
        w = w.T[0]
    w = wavTrim(w)
    yin = pyin.Model(sr)
    yin.bias = 2.0
    yin(w)

    hnmModel = hnm.Model(sr)
    hnmModel.method = method
    f0List, hFreqList, hAmpList, hPhaseList, sinusoidEnergyList, noiseEnergyList, noiseEv = hnmModel(w, yin.f0, bakedSinusoid = None)
    nHop = len(f0List)

    rs = hnmModel.synth(f0List, hFreqList, hAmpList, hPhaseList, noiseEv, sinusoidEnergyList, noiseEnergyList, noiseOn = False)
    envAnalyzer = envelope.MFIEnvelope(sr)
    envList = envAnalyzer(rs, f0List)

    db = voicedb.VoiceDB(path)
    ss = SoruceSetting(sr)
    if(type == 'v'):
        ss.allow = ('b', 'r', 'f', 'e')
    else:
        raise NotImplementedError('Only support vowel-only hone now')
    r = ss.show(w, sinusoidEnergyList, noiseEnergyList)

    f0List = f0List[r['b']:r['e']]
    hFreqList = hFreqList[r['b']:r['e']]
    hAmpList = hAmpList[r['b']:r['e']]
    hPhaseList = hPhaseList[r['b']:r['e']]
    sinusoidEnergyList = sinusoidEnergyList[r['b']:r['e']]
    noiseEnergyList = noiseEnergyList[r['b']:r['e']]
    noiseEv = noiseEv[r['b']:r['e']]
    envList = envList[r['b']:r['e']]
    nHop = r['e'] - r['b']

    rs = hnmModel.synth(f0List, hFreqList, hAmpList, hPhaseList, noiseEv, sinusoidEnergyList, noiseEnergyList, noiseOn = False)
    fd = FormantDrawing(sr)
    fd.show(rs, f0List, hFreqList, hAmpList, envList)
    formant = fd.lastCorrFormant

    phone = voicedb.Phone()
    voiced = f0List > 0.0
    phone.avgFreq = np.mean(f0List[voiced])
    phone.risePos = r['r'] - r['b']
    phone.fallPos = r['f'] - r['b']

    phone.hOFList = hFreqList.copy()
    phone.hOFList[voiced] /= f0List.reshape(nHop, 1)[voiced]
    phone.hPhaseList = hPhaseList
    phone.formantList = formant
    phone.sinusoidEnvList = envList
    phone.noiseEnvList = noiseEv
    phone.sinusoidEnergyList = sinusoidEnergyList
    phone.noiseEnergyList = noiseEnergyList
    phone.voiced = voiced

    if(db.hasAttr('phone', curr)):
        phoneList = db.getAttr('phone', curr)
    else:
        phoneList = voicedb.PhoneList('v', curr)
        db.addAttr('phone', curr, phoneList)

    phoneList.addPhone(prev, curr, next, phone)
    db.sync()

class SoruceSetting:
    def __init__(self, sr):
        self.editing = None
        self.allow = ('b', 'r', 'f', 'v', 'c', 'o', 'e')
        self.samprate = sr
        self.hopSize = self.hopSize = roundUpToPowerOf2(self.samprate * 0.005)
        self.data = {}
        self.obj = {}
        self.obj['b'] = mpl.patches.Rectangle((0.0, -1.0), 0.0, 2.0, linewidth = 0, facecolor="purple", alpha=0.25)
        self.obj['r'] = mpl.patches.Rectangle((0.0, -1.0), 0.0, 2.0, linewidth = 0, facecolor="green", alpha=0.25)
        self.obj['f'] = mpl.patches.Rectangle((0.0, -1.0), 0.0, 2.0, linewidth = 0, facecolor="green", alpha=0.25)
        self.obj['v'] = mpl.patches.Rectangle((0.0, -1.0), 0.0, 2.0, linewidth = 0, facecolor="blue", alpha=0.25)
        self.obj['c'] = mpl.patches.Rectangle((0.0, -1.0), 0.0, 2.0, linewidth = 0, facecolor="cyan", alpha=0.25)
        self.obj['o'] = mpl.patches.Rectangle((0.0, -1.0), 0.0, 2.0, linewidth = 0, facecolor="yellow", alpha=0.25)
        self.obj['e'] = mpl.patches.Rectangle((0.0, -1.0), 0.0, 2.0, linewidth = 0, facecolor="purple", alpha=0.25)

        self.name = {}
        self.name['b'] = 'begin'
        self.name['r'] = 'rise'
        self.name['f'] = 'fall'
        self.name['v'] = 'vowel'
        self.name['c'] = 'consonant'
        self.name['o'] = 'onset'
        self.name['e'] = 'end'

    def onKeyPressed(self, event):
        if(event.key == 'escape'):
            print('Change to normal.')
            self.editing = None
        elif(event.key in self.allow):
            print('Change to %s.' % (self.name[event.key]))
            self.editing = event.key

    def onPressed(self, event):
        xd, yd = event.xdata, event.ydata
        x, y = event.x, event.y

        if(xd is None or yd is None):
            return

        if(not self.editing is None):
            if(self.editing == 'b' or self.editing == 'e'):
                self.data[self.editing] = min(self.nHop, max(0, int(round(xd))))
            else:
                self.data[self.editing] = min(self.data['e'], max(self.data['b'], int(round(xd))))
            if(self.editing == 'e' or self.editing == 'f'):
                self.obj[self.editing].set_x(self.data[self.editing])
                self.obj[self.editing].set_width(self.nHop - self.data[self.editing])
            else:
                self.obj[self.editing].set_width(self.data[self.editing])
            event.canvas.draw()

    def show(self, x, sinusoidEnergyList, noiseEnergyList):
        self.nHop = len(sinusoidEnergyList)
        self.fig = pl.figure()
        self.sbp = pl.subplot(111)
        pl.plot(np.arange(len(x)) / self.hopSize, x)
        pl.plot(sinusoidEnergyList)
        pl.plot(noiseEnergyList)
        for key, rect in self.obj.items():
            if(key in self.allow):
                self.sbp.add_patch(rect)
        b, e = cutSilent(x, sinusoidEnergyList, noiseEnergyList, self.hopSize, self.samprate)
        for name in self.allow:
            if(not name in self.data):
                self.data[name] = 0
                if(name == 'b' or name == 'r'):
                    self.data[name] = b
                elif(name == 'e' or name == 'f'):
                    self.data[name] = e
            if(name == 'e' or name == 'f'):
                self.obj[name].set_x(self.data[name])
                self.obj[name].set_width(self.nHop - self.data[name])
            else:
                self.obj[name].set_width(self.data[name])
        self.fig.canvas.mpl_connect('button_press_event', self.onPressed)
        self.fig.canvas.mpl_connect('key_press_event', self.onKeyPressed)
        pl.show()
        return self.data

class FormantDrawing:
    def __init__(self, sr):
        self.samprate = sr
        self.lpcOrder = 15
        self.lpcCutoff = 7e3
        self.hopSize = roundUpToPowerOf2(self.samprate * 0.005)
        self.fftSize = roundUpToPowerOf2(self.samprate * 0.05)
        self.formantX = []
        self.formantY = []

        self.status = None

    def onPressed(self, event):
        xd, yd = event.xdata, event.ydata
        x, y = event.x, event.y
        if(xd is None or yd is None):
            return
        xHop = int(round(xd))

        if(isinstance(self.status, int)):
            iFormant = self.status
            if(event.button == 1):
                try:
                    idx = self.formantX[iFormant].index(xHop)
                    self.formantY[iFormant][idx] = yd
                    print('Replaced #%d @ (%d, %f)' % (iFormant, xHop, yd))
                except ValueError:
                    self.formantX[iFormant].append(xHop)
                    self.formantY[iFormant].append(yd)
                    print('Added #%d @ (%d, %f)' % (iFormant, xHop, yd))
                order = np.argsort(self.formantX[iFormant])
                self.formantDraw[iFormant].set_xdata(np.asarray(self.formantX[iFormant])[order])
                self.formantDraw[iFormant].set_ydata(np.asarray(self.formantY[iFormant])[order])
                event.canvas.draw()
            elif(event.button == 3):
                try:
                    idx = self.formantX[iFormant].index(xHop)
                except ValueError:
                    print('Remove #%d @ %d, not found.' % (iFormant, xHop))
                    return
                del self.formantX[iFormant][idx]
                del self.formantY[iFormant][idx]
                order = np.argsort(self.formantX[iFormant])
                self.formantDraw[iFormant].set_xdata(np.asarray(self.formantX[iFormant])[order])
                self.formantDraw[iFormant].set_ydata(np.asarray(self.formantY[iFormant])[order])
                event.canvas.draw()
                print('Removed #%d @ %d' % (iFormant, xHop))

    def onKeyPressed(self, event):
        if(event.key == 's'):
            print('Low Reso STFT')
            self.bgDraw.set_data(self.lStftMagnList.T)
            event.canvas.draw()
        elif(event.key == 'h'):
            print('High Reso STFT')
            self.bgDraw.set_data(self.hStftMagnList.T)
            event.canvas.draw()
        elif(event.key == 'e'):
            print('Envelope')
            self.bgDraw.set_data(self.envList.T)
            event.canvas.draw()
        elif(event.key == 'l'):
            print('%d-order LPC' % (self.lpcOrder))
            self.bgDraw.set_data(self.lpcMagnList.T)
            event.canvas.draw()
        elif(event.key == 'a'):
            self.formantX.append([])
            self.formantY.append([])
            iFormant = len(self.formantX) - 1
            self.formantDraw.append(pl.plot(self.formantX[iFormant], self.formantY[iFormant], marker='o')[0])
            event.canvas.draw()
            print('Added #%d' % (iFormant))
        elif(event.key == 'x'):
            if(len(self.formantX) <= 0):
                print('Nothing to remove.')
                return
            del self.formantX[-1]
            del self.formantY[-1]
            self.formantDraw[-1].remove()
            del self.formantDraw[-1]
            event.canvas.draw()
            print('Removed #%d' % (len(self.formantX)))
        elif(event.key == 'f'):
            print('Refine All...')
            self.status = None
            self.refine()
        elif(event.key.isnumeric()):
            iFormant = int(event.key)
            if(iFormant >= len(self.formantX)):
                print('No such formant. Please add.')
                self.status = None
                return
            self.status = iFormant
            print('Switched to #%d' % (iFormant))
        elif(event.key == 'escape'):
            self.status = None
            print('Switched to normal.')

    def genFormantList(self):
        nFormant = len(self.formantX)
        nHop = len(self.hStftMagnList)
        formantList = []
        formantList.append(np.full(nHop, 60.0))
        for iFormant in range(nFormant):
            if(not self.formantX[iFormant]):
                continue
            formantSeq = np.zeros(nHop)
            order = np.argsort(self.formantX[iFormant])
            fX = np.asarray(self.formantX[iFormant])[order]
            fY = np.asarray(self.formantY[iFormant])[order] / self.fftSize * self.samprate
            iplX = np.concatenate(((0.0,), fX, (nHop,)))
            iplY = np.concatenate(((fY[0],), fY, (fY[-1],)))
            formantList.append(ipl.interp1d(iplX, iplY, kind = 'linear')(np.arange(nHop)))
        if(len(formantList) == 1):
            print('No formant.')
            return
        return np.asarray(formantList)

    def refine(self):
        nFormant = len(self.formantX)
        nHop = len(self.hStftMagnList)
        formantList = self.genFormantList()
        if(formantList is None):
            return
        formant = np.zeros((3, len(formantList), nHop))
        formant[0] = formantList
        formant[1] = 250.0
        formant[2] = 1.0
        formant = formant.transpose(2, 1, 0)
        corr = correcter.Processor(self.samprate)
        formant = corr.seq(self.hFreqList, self.hAmpList, self.envList, formant, trustAmp = False)
        self.lastCorrFormant = formant
        formant = formant.transpose(2, 1, 0)
        for draw in self.formantDraw:
            draw.remove()
        self.formantX = []
        self.formantY = []
        self.formantDraw = []
        for iFormant, f in enumerate(formant[0][1:]):
            need = f != 0
            self.formantX.append(list(np.arange(nHop)[need]))
            self.formantY.append(list(f[need] / self.samprate * self.fftSize))
            self.formantDraw.append(pl.plot(np.asarray(self.formantX[iFormant]), np.asarray(self.formantY[iFormant]), marker='o')[0])
        self.fig.canvas.draw()

    def show(self, x, f0List, hFreqList, hAmpList, envList):
        self.fig = pl.figure()
        self.sbp = pl.subplot(111)

        stftAnalyzer = stft.AdaptiveAnalyzer(self.fftSize, self.hopSize, self.samprate, window = 'blackman')
        hStftMagnList, _ = stftAnalyzer(x, f0List)
        lStftMagnList, _ = stftAnalyzer(x, f0List, BFac = 0.5)

        lpcAnalyzer = lpc.Burg(self.lpcOrder, f0List, self.samprate, resample = self.lpcCutoff * 2)
        lpcAnalyzer.hopSize = self.hopSize
        lpcAnalyzer(x)
        lpcMagnList = np.zeros(hStftMagnList.shape)
        lpcSpectrumSize = int(round(self.lpcCutoff * 2 / self.samprate * self.fftSize))
        lpcMagnList.T[:lpcSpectrumSize // 2 + 1] = lpcAnalyzer.toSpectrum(lpcSpectrumSize).T

        self.hStftMagnList = hStftMagnList
        self.lStftMagnList = lStftMagnList
        self.lpcMagnList = lpcMagnList
        self.bgDraw = pl.imshow(self.lStftMagnList.T, origin = 'lower', cmap = 'jet', interpolation = 'bicubic', aspect = 'auto')
        self.formantDraw = []
        for iFormant in range(len(self.formantX)):
            order = np.argsort(self.formantX[iFormant])
            self.formantDraw.append(pl.plot(np.asarray(self.formantX[iFormant])[order], np.asarray(self.formantY[iFormant])[order], marker='o')[0])

        self.fig.canvas.mpl_connect('button_press_event', self.onPressed)
        self.fig.canvas.mpl_connect('key_press_event', self.onKeyPressed)

        self.hFreqList = hFreqList
        self.hAmpList = hAmpList
        self.envList = envList

        pl.show()
        return self.genFormantList()[1:].T
