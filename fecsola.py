import numpy as np
import scipy.signal as sp
import scipy.interpolate as ipl
import pylab as pl
from common import *
import stft
import lpc

# Klatt Combination FECSOLA
class Processor:
    def __init__(self, sr):
        self.samprate = sr
        self.hfFreq = 6000.0
        self.hfBw = 125.0

    def decomp(self, env, formant):
        nSpec = len(env)
        nFFT = (nSpec - 1) * 2
        nFormant = formant.shape[0]
        nComp = nFormant + 1
        fRange = np.arange(nSpec) / nFFT * self.samprate

        # formant
        combine = np.zeros(nSpec)
        compList = np.zeros((nComp, nSpec))
        for iFormant, (F, bw, amp) in enumerate(formant):
            iComp = iFormant
            comp = klattFilter(fRange, np.power(10, amp / 20.0), F, bw, self.samprate)
            pl.plot()
            combine += comp
            compList[iComp] = comp
        # hf
        hfFreqBin = self.hfFreq / self.samprate * nFFT
        hfFilter = klattFilter(fRange, 1.0, self.hfFreq, self.hfBw, self.samprate)
        hfFilter[int(np.ceil(hfFreqBin)):] = 1.0
        compList[-1] = hfFilter
        compList[-1] *= env
        combine += compList[-1]
        del hfFreqBin, comp
        compList = np.clip(compList, 1e-30, np.inf)
        normalized = env / combine
        compFilterList = compList.copy()
        for iComp in range(nComp):
            compList[iComp] *= normalized

        return hfFilter, compList, compFilterList, combine, normalized

    def transform(self, origFormant, newFormant, compList, compFilterList, hfFilter, hfMove, hfAmp):
        nSpec = compList.shape[1]
        nFFT = (nSpec - 1) * 2
        nFormant = origFormant.shape[0]
        nComp = nFormant + 1
        fRange = np.arange(nSpec) / nFFT * self.samprate

        compList = compList.copy()
        for iFormant in range(nFormant):
            iComp = iFormant
            origF, origBw, origAmp = origFormant[iFormant]
            newF, newBw, newAmp = newFormant[iFormant]
            if((origFormant[iFormant] != newFormant[iFormant]).any()):
                if(origF - origBw <= 0.0 or newF - newBw <= 0.0):
                    origKey = [origF - origBw]
                    newKey = [newF - newBw]
                else:
                    origKey = [0.0, origF - origBw]
                    newKey = [0.0, newF - newBw]
                origKey.append(origF)
                origKey.append(origF + origBw)
                newKey.append(newF)
                newKey.append(newF + newBw)
                if(newF + newBw < fRange[-1] or origF + origBw < fRange[-1]):
                    origKey.append(fRange[-1])
                    newKey.append(fRange[-1])
                nRange = ipl.Akima1DInterpolator(origKey, newKey)(fRange)
                newTrans = ipl.interp1d(nRange, compFilterList[iComp], bounds_error = False, fill_value = compFilterList[iComp][-1],kind='linear')(fRange)
                newComp = ipl.interp1d(nRange, compList[iComp], bounds_error = False, fill_value = compList[iComp][-1], kind='linear')(fRange)
                newFilter = klattFilter(fRange, np.power(10, newAmp / 20.0), newF, newBw, self.samprate)
                diffFac = newFilter / newTrans
                compList[iComp] = newComp * diffFac
        # non-linear transform for hf
        if(hfMove != 0.0 or hfAmp != 1.0):
            iComp = -1
            origKey = (0.0, self.hfFreq - self.hfBw, self.hfFreq, self.hfFreq + self.hfBw, fRange[-1])
            newKey = (0.0, self.hfFreq - self.hfBw + hfMove, self.hfFreq + hfMove, self.hfFreq + self.hfBw + hfMove, fRange[-1])
            nRange = ipl.Akima1DInterpolator(origKey, newKey)(fRange)
            newTrans = ipl.interp1d(nRange, compFilterList[iComp], bounds_error = False, fill_value = compFilterList[iComp][-1], kind='linear')(fRange)
            newComp = ipl.interp1d(nRange, compList[iComp], bounds_error = False, fill_value = compList[iComp][-1], kind='linear')(fRange)
            newFilter = klattFilter(fRange, hfAmp, self.hfFreq + hfMove, self.hfBw, self.samprate)
            hfFreqBin = (self.hfFreq + hfMove) / self.samprate * nFFT
            newFilter[int(np.ceil(hfFreqBin)):] = hfAmp
            diffFac = newFilter / hfFilter
            compList[-1] = newComp * diffFac
        return compList

    def mix(self, origEnv, origFormant, newEnv, newFormant, ratio, hfMove = 0.0, hfAmp = 1.0):
        nSpec = len(origEnv)
        nFFT = (nSpec - 1) * 2
        nFormant = origFormant.shape[0]
        nComp = nFormant + 1
        assert(origFormant.shape[0] == newFormant.shape[0])

        fRange = np.arange(nSpec) / nFFT * self.samprate

        # to linear space
        origEnv = np.power(10, origEnv / 20.0)
        newEnv = np.power(10, newEnv / 20.0)

        # decomp
        hfFilter, origCompList, origCompFilterList, _, origNormalized = self.decomp(origEnv, origFormant)
        _, newCompList, newCompFilterList, _, newNormalized = self.decomp(newEnv, newFormant)

        # mix
        # recomp
        normalized = origNormalized + (newNormalized - origNormalized) * ratio
        compList = origCompList.copy()
        for iFormant in range(nFormant):
            iComp = iFormant
            compList[iComp] = origCompFilterList[iComp] * normalized
        compList[-1] = origCompList[-1] + (newCompList[-1] - origCompList[-1]) * ratio
        # hf
        compFilterList = origCompFilterList.copy()
        compFilterList[-1] = origCompFilterList[-1] + (newCompFilterList[-1] - origCompFilterList[-1]) * ratio
        # formant
        mixedFormant = origFormant + (newFormant - origFormant) * ratio

        # non-linear transform for formants
        compList = self.transform(origFormant, mixedFormant, compList, compFilterList, hfFilter, hfMove, hfAmp)

        # comp
        mixedEnv = np.zeros(nSpec)
        for iComp in range(nComp):
            mixedEnv += compList[iComp]

        return np.log10(mixedEnv) * 20.0

    def move(self, origEnv, origFormant, newFormant, hfMove = 0.0, hfAmp = 1.0):
        nSpec = len(origEnv)
        nFFT = (nSpec - 1) * 2
        nFormant = origFormant.shape[0]
        nComp = nFormant + 1
        assert(origFormant.shape[0] == newFormant.shape[0])

        fRange = np.arange(nSpec) / nFFT * self.samprate

        # to linear space
        origEnv = np.power(10, origEnv / 20.0)

        # decomp
        hfFilter, compList, compFilterList, _, _ = self.decomp(origEnv, origFormant)

        # non-linear transform for formants
        compList = self.transform(origFormant, newFormant, compList, compFilterList, hfFilter, hfMove, hfAmp)
        #pl.plot(np.log10(compList[iComp]) * 20.0, label = 'Comp %d' % (iComp))
        #pl.plot(np.log10(compFilterList[iComp]) * 20.0, label = 'Filter %d' % (iComp))
        #pl.show()
        # comp
        newEnv = np.zeros(nSpec)
        for iComp in range(nComp):
            newEnv += compList[iComp]
        '''
        pl.plot(np.log10(origEnv) * 20.0, label = 'Comp %d' % (iComp))
        pl.plot(np.log10(newEnv) * 20.0, label = 'Filter %d' % (iComp))
        pl.show()
        '''
        return np.log10(newEnv) * 20.0
