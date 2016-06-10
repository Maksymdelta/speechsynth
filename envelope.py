import numpy as np
import scipy.signal as sp
import scipy.interpolate as ipl
import pylab as pl
from common import *
import stft
import lpc
import hnm
import numba as nb

class TrueEnvelope:
    def __init__(self, sr):
        self.samprate = sr
        self.hopSize = roundUpToPowerOf2(self.samprate * 0.005)
        self.fftSize = roundUpToPowerOf2(self.samprate * 0.05)

    def __call__(self, x, f0List):
        nX = len(x)
        nHop = getNFrame(nX, self.hopSize)
        nSpec = int(self.fftSize / 2) + 1
        assert(len(f0List) == nHop)
        stftAnalyzer = stft.AdaptiveAnalyzer(self.fftSize, self.hopSize, self.samprate, window = 'blackman')
        magnList, phaseList = stftAnalyzer(x, f0List)

        envList = np.zeros((nHop, nSpec))
        for iHop, f0 in enumerate(f0List):
            order = int(72 * (self.samprate / 44100)) if(f0 <= 0.0) else int(round(self.samprate / (2.0 * f0)))
            envList[iHop] = trueEnv(magnList[iHop], order, iterCount = 128, maxStep = 1.0)
            '''if(iHop == 389):
                pl.plot(magnList[iHop])
                pl.plot(envList[iHop])
                pl.show()'''
        #pl.imshow(envList.T, origin = 'lower', cmap = 'jet', interpolation = 'bicubic', aspect = 'auto')
        #pl.show()
        return envList

class MFIEnvelope:
    def __init__(self, sr):
        self.samprate = sr
        self.hopSize = roundUpToPowerOf2(self.samprate * 0.005)
        self.fftSize = roundUpToPowerOf2(self.samprate * 0.05)
        self.preEmphasisFreq = 50.0

    def __call__(self, x, f0List):
        return self.__process__(x, f0List, self.hopSize, self.fftSize, self.samprate, self.preEmphasisFreq)

    @staticmethod
    def __process__(x, f0List, hopSize, fftSize, samprate, preEmphasisFreq):
        nX = len(x)
        nHop = getNFrame(nX, hopSize)
        nSpec = int(fftSize / 2) + 1
        nyq = samprate / 2.0
        assert(len(f0List) == nHop)

        pe = preEmphasisResponse(np.arange(nSpec) / fftSize * samprate, preEmphasisFreq, samprate)
        x = preEmphasis(x, preEmphasisFreq, samprate)
        stftAnalyzer = stft.AdaptiveAnalyzer(fftSize, hopSize, samprate, window = 'blackman')
        magnList, phaseList = stftAnalyzer(x, f0List)

        envList = np.zeros((nHop, nSpec))
        filter = sp.firwin(21, 0.06, window='hanning', pass_zero=True)
        w, h = sp.freqz(filter, worN = 2049)
        for iHop, f0 in enumerate(f0List):
            rf0 = f0
            if(f0 <= 0.0):
                f0 = 256.0
            iCenter = iHop * hopSize
            maxOffset = int(np.ceil(samprate / (2 * f0)))
            stdev = samprate / (3 * f0)
            windowSize = int(2 * samprate / f0)
            if(windowSize % 2 != 0):
                windowSize += 1
            halfWindowSize = int(windowSize / 2)
            window = sp.gaussian(windowSize, stdev)
            window = window / np.sqrt(np.mean(window ** 2))
            window *= 2 / np.sum(window)
            maxSpec = np.full(nSpec, -np.inf)
            minSpec = np.full(nSpec, np.inf)
            igSpec = np.zeros(nSpec)
            for offset in range(-maxOffset, maxOffset):
                frame = getFrame(x, iCenter + offset, windowSize) * window
                padded = np.zeros(fftSize)
                padded[:halfWindowSize] = frame[halfWindowSize:]
                padded[-halfWindowSize:] = frame[:halfWindowSize]
                spec = np.abs(np.fft.rfft(padded))
                need = maxSpec < spec
                maxSpec[need] = spec[need]
                need = spec < minSpec
                minSpec[need] = spec[need]
                igSpec += spec
            if((maxSpec == 0.0).all() or (minSpec == 0.0).all()):
                envList[iHop] = -np.inf
                continue
            igSpec /= 2 * maxOffset
            np.log10(igSpec, out = igSpec)
            igSpec *= 20.0
            smoothSpec = np.convolve(igSpec, filter)[10:-10]
            smoothSpec[:10] = igSpec[:10]
            smoothSpec[-10:] = igSpec[-10:]
            smoothSpec = toLog(toLinear(smoothSpec) * (np.sum(toLinear(igSpec)) / np.sum(toLinear(smoothSpec))))
            '''if(iHop == 389):
                pl.plot(magnList[iHop])
                avg = np.log10((minSpec + maxSpec) / 2.0) * 20.0
                avgSpec = np.convolve(avg, filter)[10:-10]
                avgSpec[:10] = avg[:10]
                avgSpec[-10:] = avg[-10:]
                pl.plot(avgSpec)
                pl.plot(igSpec)
                pl.plot(smoothSpec)
                pl.show()'''
            envList[iHop] = smoothSpec
            envList[iHop] = toLog(toLinear(envList[iHop]) / pe)
        #pl.imshow(envList.T, origin = 'lower', cmap = 'jet', interpolation = 'bicubic', aspect = 'auto')
        #pl.show()
        return envList
