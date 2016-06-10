import numpy as np
import scipy.signal as sp
import pylab as pl
import stft
from common import *

class Model:
    def __init__(self, f0List, sr):
        self.samprate = sr
        self.hopSize = roundUpToPowerOf2(self.samprate * 0.005)
        self.fftSize = self.fftSize = roundUpToPowerOf2(self.samprate * 0.025)
        self.lpfCutoff = 5e3
        self.f0List = f0List.copy()
        self.pulseList = []
        
    def analyze(self, x):
        nHop = len(self.f0List)
        halfHopSize = int(self.hopSize / 2)
        nX = len(x)
        nyq = self.samprate / 2.0
        filtOrder = int(np.ceil(self.hopSize * 0.4))
        if(filtOrder % 2 == 0):
            filtOrder += 1
        
        # refine f0
        stftAnalyzer = stft.AdaptiveAnalyzer(self.fftSize, self.hopSize, self.samprate)
        magnList, phaseList, self.f0List = stftAnalyzer(x, self.f0List, refineF0 = True)
        
        # apply lpf and remove dc offset
        filter = sp.firwin(filtOrder, self.lpfCutoff / nyq, pass_zero = True)
        fx = sp.fftconvolve(x, filter)[int((filtOrder - 1) / 2):-int((filtOrder - 1) / 2)]
        fx -= np.mean(fx)
        
        # extract voice pulses
        pulseList = []
        iCenter = 0
        globPeak = np.max(np.abs(fx))
        addedRight = 0
        while(True):
            # get first voiced frame after iSample
            iHop = int(np.ceil(iCenter / self.hopSize))
            voiced = np.arange(iHop, nHop)[self.f0List[iHop:] > 0.0]
            if(len(voiced) == 0):
                break
            iHop = voiced[0]
            iCenter = iHop * self.hopSize
            iLeft = iCenter - halfHopSize
            iRight = iCenter + halfHopSize
            f0 = self.f0List[iHop]
            
            # find extremum
            width = int(np.ceil(self.samprate / f0))
            frame = getFrame(fx, iCenter, width)
            iMax = np.argmax(frame)
            iMin = np.argmin(frame)
            if(frame[iMax] == frame[iMin]):
                iEx = iCenter
            else:
                iEx = iCenter - int(width / 2) + (iMax if(frame[iMax] > frame[iMin]) else iMin)
            iEx = parabolicInterpolation(fx, iEx, val = False, overAdjust = False)
            addPoint(pulseList, iEx, 0.3 * self.samprate / f0)
            # do correlation
            iSaveEx = iEx
            while(True):
                iHop = int(np.ceil(iCenter / self.hopSize))
                f0 = self.f0List[iHop]
                if(f0 <= 0.0):
                    break
                iEx, peak, bestCorr = self.correlateExtremum(fx, iEx, int(np.ceil(self.samprate / f0)), iEx - int(1.25 * self.samprate / f0), iEx - int(np.ceil(0.8 * self.samprate / f0)))
                if(bestCorr == -np.inf):
                    iEx -= self.samprate / f0
                if(iEx < iLeft):
                    if(bestCorr > 0.7 and peak > 0.02 * globPeak and iEx - addedRight > 0.8 * self.samprate / f0):
                        addPoint(pulseList, iEx, 0.3 * self.samprate / f0)
                    break
                if(bestCorr > 0.3 and (peak == 0.0 or peak > 0.01 * globPeak)):
                    if(iEx - addedRight > 0.8 * self.samprate / f0):
                        addPoint(pulseList, iEx, 0.3 * self.samprate / f0)
            iEx = iSaveEx
            while(True):
                iHop = int(np.ceil(iCenter / self.hopSize))
                f0 = self.f0List[iHop]
                if(f0 <= 0.0):
                    break
                iEx, peak, bestCorr = self.correlateExtremum(fx, iEx, int(np.ceil(self.samprate / f0)), iEx + int(0.8 * self.samprate / f0), iEx + int(np.ceil(1.25 * self.samprate / f0)))
                if(bestCorr == -np.inf):
                    iEx += int(round(self.samprate / f0))
                if(iEx > iRight):
                    if(bestCorr > 0.7 and peak > 0.02 * globPeak):
                        addPoint(pulseList, iEx, 0.3 * self.samprate / f0)
                        addedRight = iEx
                    break
                if(bestCorr > 0.3 and (peak == 0.0 or peak > 0.01 * globPeak)):
                    addPoint(pulseList, iEx, 0.3 * self.samprate / f0)
                    addedRight = iEx
            
            iCenter = iRight
        self.pulseList = np.asarray(pulseList)
        self.x = x.copy()
    
    @staticmethod
    def correlateExtremum(x, iEx, windowSize, minIdx, maxIdx):
        iEx = int(round(iEx))
        if(windowSize % 2 == 1):
            windowSize += 1
        iBest = iEx
        peak = 0
        bestCorr = -np.inf
        a = getFrame(x, iEx, windowSize)
        energyA = np.sum(a * a)
        r1, r2, r3 = 0.0, 0.0, 0.0
        for iCand in range(int(minIdx), int(np.ceil(maxIdx))):
            b = getFrame(x, iCand, windowSize)
            energyB = np.sum(b * b)
            corr = np.sum(a * b)
            localPeak = np.max(np.abs(b))
            r1, r2 = r2, r3
            r3 = corr / np.sqrt(energyA * energyB) if(corr != 0) else 0.0
            if(r2 > bestCorr and r2 >= r1 and r2 >= r3):
                bestR1 = r1
                bestCorr = r2
                bestR3 = r3
                iBest = iCand
                peak = localPeak
        iBest = parabolicInterpolation(x, iBest, val = False, overAdjust = False)
        return iBest, peak, bestCorr