import numpy as np
import scipy.signal as sp
import pylab as pl
from common import *

class AdaptiveAnalyzer:
    windows = {
        'hanning': (sp.hanning, 1.5),
        'blackman': (sp.blackman, 1.73),
        'blackmanharris': (sp.blackmanharris, 2.0),
        'gaussian': (gaussianWindow, 3.25)
    }

    def __init__(self, fftSize, hopSize, sr, window = 'blackman'):
        self.hopSize = hopSize
        self.fftSize = fftSize
        self.samprate = sr
        self.window = window

    def __call__(self, input, f0List, refineF0 = False, rmDC = True, energy = False, BFac = 1.0):
        nHop = len(f0List)
        nSpec = int(self.fftSize / 2) + 1
        magnList = np.zeros((len(f0List), nSpec))
        phaseList = np.zeros((len(f0List), nSpec))
        if(refineF0):
            rf0List = np.zeros(nHop)
        if(energy):
            energyList = np.zeros(nHop)
        windowFunc, B = self.windows[self.window]
        B *= BFac

        for iHop, f0 in enumerate(f0List):
            if(f0 > 0.0):
                windowSize = int(min(self.fftSize, np.ceil(self.samprate / f0) * B * 2.0))
                if(windowSize % 2 != 0):
                    windowSize += 1
                halfWindowSize = int(windowSize / 2)
            else:
                windowSize = self.hopSize * 2
                halfWindowSize = self.hopSize
            window = windowFunc(windowSize)
            windowNormFac = 2.0 / np.sum(window)
            iCenter = self.hopSize * iHop

            if(rmDC):
                frame = getFrame(input, iCenter, windowSize)
                frame -= np.mean(frame)
                frame *= window

            if(energy):
                energyList[iHop] = np.sqrt(np.mean((frame * (halfWindowSize * windowNormFac)) ** 2))

            # padding and fftshift
            x = np.zeros(self.fftSize)
            x[:halfWindowSize] = frame[halfWindowSize:]
            x[-halfWindowSize:] = frame[:halfWindowSize]

            ffted = np.fft.rfft(x)
            magnList[iHop] = np.log10(np.clip(np.abs(ffted) * windowNormFac, 1e-10, np.inf)) * 20.0
            phaseList[iHop] = np.unwrap(np.angle(ffted))

            if(f0 > 0.0 and refineF0):
                frame = getFrame(input, iCenter - 1, windowSize)
                frame -= np.mean(frame)
                frame *= window
                x = np.zeros(self.fftSize)
                x[:halfWindowSize] = frame[halfWindowSize:]
                x[-halfWindowSize:] = frame[:halfWindowSize]
                lowerIdx = int(np.floor(f0 * self.fftSize / self.samprate  * 0.7))
                upperIdx = int(np.ceil(f0 * self.fftSize / self.samprate * 1.3))
                peakIdx = np.argmax(magnList[iHop][lowerIdx:upperIdx]) + lowerIdx
                phase = phaseList[iHop][peakIdx]
                dPhase = np.unwrap(np.angle(np.fft.rfft(x)))[peakIdx]

                phase -= np.floor(phase / 2.0 / np.pi) * 2.0 * np.pi
                dPhase -= np.floor(dPhase / 2.0 / np.pi) * 2.0 * np.pi
                if(phase < dPhase):
                    phase += 2 * np.pi
                rf0 = (phase - dPhase) / 2.0 / np.pi * self.samprate
                if(np.abs(rf0 / self.samprate * self.fftSize - peakIdx) > 1.0 or np.abs(rf0 - f0) > 10.0):
                    rf0List[iHop] = f0
                else:
                    rf0List[iHop] = rf0
        ret = [magnList, phaseList]

        if(refineF0):
            ret.append(rf0List)
        if(energy):
            ret.append(energyList)

        return tuple(ret)
