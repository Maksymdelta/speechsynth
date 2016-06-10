import numpy as np
import pylab as pl
from common import *
import matplotlib as mpl

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

def assemblyFilter(freq, amp, bw, nfft, sr):
    nSpec = int(nfft / 2) + 1
    o = np.zeros(nSpec)
    X = np.arange(nSpec) / nfft * sr
    for i in range(len(freq)):
        o += klattFilter(X, amp[i], freq[i], bw[i], sr)
    return o

class UI():
    def __init__(self, spec, sr):
        self.sr = sr
        self.spec = spec
        self.cfs = []
        self.bws = []
        self.amps = []
        self.objs = []
        self.nfft = (len(spec) - 1) * 2
        self.X = np.arange(len(spec)) / self.nfft * sr
        self.pressing = -1
        self.lastLoc = (-1, -1)

    def getIdxOnLoc(self, x, y, xd, yd):
        for i, freq in enumerate(self.cfs):
            amp = self.amps[i]
            tx, ty = self.sbp.transData.transform((freq, amp))
            if(np.abs(x - tx) < 4.0 and np.abs(y - ty) < 4.0):
                return i
        return -1

    def onKeyPressed(self, event):
        if(event.key == '-' and self.pressing != -1):
            d = self.pressing
            self.pressing = -1
            self.cfs = np.delete(self.cfs, d)
            self.amps = np.delete(self.amps, d)
            self.bws = np.delete(self.bws, d)
            self.objs[d][0].remove()
            self.objs[d][1].remove()
            del self.objs[d]

            self.mixtureFilterPlot.set_ydata(np.log10(assemblyFilter(self.cfs, np.power(10, self.amps / 20.0), self.bws, self.nfft, self.sr)) * 20.0)
            self.fig.canvas.draw()
        elif((event.key == '=' or event.key == '+') and self.pressing == -1):
            xd, yd = self.lastLoc
            yd = np.clip(yd, -120.0, -1e-2)
            self.cfs = np.concatenate((self.cfs, np.array((xd,))))
            self.amps = np.concatenate((self.amps, np.array((yd,))))
            self.bws = np.concatenate((self.bws, np.array((200.0,))))

            f = klattFilter(self.X, np.power(10, self.amps[-1] / 20.0), self.cfs[-1], self.bws[-1], self.sr)
            l = pl.plot(self.X, np.log10(f) * 20.0, 'g-')[0]
            o = pl.plot((self.cfs[-1]), (self.amps[-1]), 'ro')[0]
            self.objs.append((l, o))

            self.mixtureFilterPlot.set_ydata(np.log10(assemblyFilter(self.cfs, np.power(10, self.amps / 20.0), self.bws, self.nfft, self.sr)) * 20.0)

            self.fig.canvas.draw()

    def onPressed(self, event):
        xd, yd = event.xdata, event.ydata
        x, y = event.x, event.y

        if(xd is None or yd is None):
            return

        self.lastLoc = (xd, yd)
        self.pressing = self.getIdxOnLoc(x, y, xd, yd)

    def onReleased(self, event):
        self.pressing = -1

    def onMoved(self, event):
        xd, yd = event.xdata, event.ydata
        x, y = event.x, event.y

        if(xd is None or yd is None):
            return

        deltaLoc = (xd - self.lastLoc[0], yd - self.lastLoc[1])
        self.lastLoc = (xd, yd)

        if(self.pressing >= 0):
            idx = self.pressing
            if(event.button == 1):
                self.cfs[idx] = np.clip(xd, 50.0, 18000.0)
                self.amps[idx] = np.clip(yd, -120.0, -1e-2)
            elif(event.button == 3):
                self.bws[idx] = np.clip(self.bws[idx] + deltaLoc[0], 25.0, 2500.0)

            f = klattFilter(self.X, np.power(10, self.amps[idx] / 20.0), self.cfs[idx], self.bws[idx], self.sr)

            self.objs[idx][0].set_ydata(np.log10(f) * 20.0)
            self.objs[idx][1].set_xdata((self.cfs[idx]))
            self.objs[idx][1].set_ydata((self.amps[idx]))

            self.mixtureFilterPlot.set_ydata(np.log10(assemblyFilter(self.cfs, np.power(10, self.amps / 20.0), self.bws, self.nfft, self.sr)) * 20.0)

            self.fig.canvas.draw()

    def show(self):
        self.cfs = np.array(self.cfs)
        self.amps = np.array(self.amps)
        self.bws = np.array(self.bws)
        if(self.cfs.shape != self.amps.shape != self.bws.shape):
            raise ValueError("Bad input value.")

        self.fig = pl.figure()
        self.sbp = pl.subplot(111)
        pl.plot(self.X, self.spec)

        self.sbp.set_xlim(0, 22050)
        self.sbp.set_ylim(-90, 0)

        self.objs = []
        for i in range(len(self.cfs)):
            f = klattFilter(self.X, np.power(10, self.amps[i] / 20.0), self.cfs[i], self.bws[i], self.sr)
            l = pl.plot(self.X, np.log10(f) * 20.0, 'g-')[0]
            o = pl.plot((self.cfs[i]), (self.amps[i]), 'ro')[0]
            self.objs.append((l, o))

        self.mixtureFilterPlot = pl.plot(self.X, np.log10(assemblyFilter(self.cfs, np.power(10, self.amps / 20.0), self.bws, self.nfft, self.sr)) * 20.0, 'r-')[0]

        self.fig.canvas.mpl_connect('button_press_event', self.onPressed)
        self.fig.canvas.mpl_connect('button_release_event', self.onReleased)
        self.fig.canvas.mpl_connect('key_press_event', self.onKeyPressed)
        self.fig.canvas.mpl_connect('motion_notify_event', self.onMoved)
        pl.show()

        self.fig = None
        self.objs = []

        order = np.argsort(self.cfs)
        self.cfs = np.array(self.cfs)[order]
        self.amps = np.array(self.amps)[order]
        self.bws = np.array(self.bws)[order]

        return self.cfs, self.bws, self.amps

def quickCreate(spec, nFormant, sr, env = None):
    nFFT = (len(spec) - 1) * 2
    ui = UI(spec, sr)
    F = formantFreq(np.arange(1, nFormant + 1))
    bw = np.full(nFormant, 200.0)
    if(env is None):
        amp = np.log10(np.power(10, spec / 20.0)[np.round(F / sr * nFFT).astype(int)]) * 20.0
    else:
        amp = np.log10(np.power(10, env / 20.0)[np.round(F / sr * nFFT).astype(int)]) * 20.0
    ui.cfs = F
    ui.bws = bw
    ui.amps = amp
    return ui
