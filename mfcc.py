import numpy as np
import scipy.signal as sp
import scipy.interpolate as ipl
import scipy.fftpack as sfft
import pylab as pl
import stft
from common import *

class Analyzer:
    def __init__(self, sr, window = 'blackman'):
        self.samprate = sr
        self.hopSize = roundUpToPowerOf2(self.samprate * 0.005)
        self.fftSize = roundUpToPowerOf2(self.samprate * 0.05)
        self.window = window
        self.firstFilter = 0.0
        self.filterDistance = 25.0
        self.preEmphasisFreq = 50.0

    def __call__(self, x, f0List, order):
        nHop = len(f0List)
        nSpec = int(self.fftSize / 2) + 1
        nyq = self.samprate / 2

        maxMel = freqToMel(nyq)
        nFilter = int(np.ceil((maxMel - self.firstFilter) / self.filterDistance)) + 1
        fRange = np.fft.rfftfreq(self.fftSize, 1.0 / self.samprate)

        if(not self.preEmphasisFreq is None):
            x = preEmphasis(x, self.preEmphasisFreq, self.samprate)

        stftAnalyzer = stft.AdaptiveAnalyzer(self.fftSize, self.hopSize, self.samprate, window = self.window)
        magnList, _ = stftAnalyzer(x, f0List)
        magnList = np.power(10, magnList / 20.0)
        filterBank = np.zeros((nFilter, magnList.shape[1]))
        for iFilter in range(nFilter):
            peakMel = iFilter * self.filterDistance + self.firstFilter
            lFreq = melToFreq(peakMel - self.filterDistance)
            peakFreq = melToFreq(peakMel)
            rFreq = melToFreq(peakMel + self.filterDistance)
            iplX = (lFreq, peakFreq, rFreq)
            iplY = (0.0, 1.0, 0.0)
            filterBank[iFilter] = ipl.interp1d(iplX, iplY, kind = 'linear', bounds_error = False, fill_value = 0.0)(fRange)
            filterBank[iFilter] /= (rFreq - lFreq) / self.samprate * self.fftSize

        mfccList = np.zeros((nHop, order))
        for iHop in range(nHop):
            coeff = np.zeros(nFilter)
            for iFilter in range(nFilter):
                coeff[iFilter] = np.sum(magnList[iHop] * filterBank[iFilter])
            coeff = np.log10(coeff) * 20.0
            if(nFilter >= order):
                mfccList[iHop] = sfft.dct(coeff)[:order]
            else:
                mfccList[iHop][:nFilter] = sfft.dct(coeff)
        mfccList /= np.max(np.abs(mfccList))
        return mfccList

    def toLinearSpectrum(self, mfccList):
        nHop = len(mfccList)
        nSpec = int(self.fftSize / 2) + 1
        nyq = self.samprate / 2
        maxMel = freqToMel(nyq)
        nFilter = int(np.ceil((maxMel - self.firstFilter) / self.filterDistance)) + 1
        fRange = np.fft.rfftfreq(self.fftSize, 1.0 / self.samprate)

        melX = np.zeros(nFilter)
        for iFilter in range(nFilter):
            melX[iFilter] = iFilter * self.filterDistance + self.firstFilter
        melY = ipl.interp1d(np.arange(nFilter) / (nFilter - 1), melX, kind = 'linear')(np.arange(self.fftSize // 2 + 1) / (self.fftSize // 2))
        linearX = melToFreq(melY)
        mfccEnv = np.zeros((nHop, nSpec))
        for iHop in range(nHop):
            mfccEnv[iHop] = sfft.idct(mfccList[iHop], n = self.fftSize // 2 + 1)
            mfccEnv[iHop] = ipl.interp1d(linearX, mfccEnv[iHop], kind = 'linear')(fRange)
        return mfccEnv
