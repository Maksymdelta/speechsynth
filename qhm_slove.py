import numpy as np
import scipy.signal as sp
import scipy.linalg as sla
import pylab as pl
from common import *

# x: signal frame, [2N+1x1]
# fk_hat: estimate of the frequencies, [Kx1]
# win: window function, [2N+1x1]
# fs: sampling frequency, [1x1]
# iter: number of iterations, [1x1]

class QHM_AIR:
    windows = {
        'hanning': (sp.hanning, 1.6),
        'blackman': (sp.blackman, 1.73)
    }
    
    def __init__(self, f0List, sr, window = 'blackman'):
        self.samprate = sr
        self.f0List = f0List.copy()
        self.mvf = min(21e3, sr / 2 * 0.92)
        self.maxAIRIter = 10
        self.mainCorrIter = 4
        self.mainTargetSRER = 36.0
        self.extraCorrIter = 2
        self.hopSize = roundUpToPowerOf2(self.samprate * 0.005)
        self.maxHar = 192
        self.window = window
    
    def __call__(self, x):
        nyq = self.samprate / 2
        nX = len(x)
        nHop = getNFrame(nX, self.hopSize)
        assert(len(self.f0List) == nHop)
        
        windowFunc, B = self.windows[self.window]
        sWindow = sp.hanning(self.hopSize * 2)
        sRange = np.arange(-self.hopSize, self.hopSize)
        
        # air f0 refinement
        print("AIR")
        for iHop, f0 in enumerate(self.f0List):
            if(f0 <= 0.0):
                continue
            print(iHop, "/", nHop)
            windowSize = int(np.ceil(self.samprate / f0 * B * 2.0))
            if(windowSize % 2 == 0):
                windowSize += 1
            aWindow = windowFunc(windowSize)
            iCenter = self.hopSize * iHop
            nHar = int(self.mvf / f0)
            lastSRER = -np.inf
            cFrame = getFrame(x, iCenter, self.hopSize * 2) * sWindow # comparison frame
            
            for iIter in range(self.maxAIRIter):
                aFrame = getFrame(x, iCenter, windowSize) # analysis frame
                ak, bk, fk_hat, _ = self.slove(aFrame, aWindow, f0, self.mvf, self.samprate, nIter = 0, maxCorr = min(20.0, 0.12 * f0), lastFreqCorr = True)
                maxNHar = int(self.mvf / f0)
                sFrame = self.synthFromAK(ak, fk_hat, sRange, self.samprate) * sWindow
                currSRER = self.calcSRER(cFrame, sFrame)
                
                if(currSRER - lastSRER > 0.0):
                    nMean = min(8, int(nHar / 3))
                    f0 = np.mean(fk_hat.reshape(nHar * 2 + 1)[nHar + 1:nHar + 1 + nMean] / np.arange(1, nMean + 1))
                    self.f0List[iHop] = f0
                   
                    windowSize = int(np.ceil(np.ceil(self.samprate / f0) * B * 2.0))
                    if(windowSize % 2 == 0):
                        windowSize += 1
                    aWindow = windowFunc(windowSize)
                    nHar = int(self.mvf / f0)
                if(currSRER - lastSRER < 0.1):
                    break
                lastSRER = currSRER

            print("    SRER:", currSRER)
        del iHop, f0, windowSize, aWindow, iCenter, nHar, lastSRER, cFrame, iIter, aFrame, ak, bk, fk_hat, sFrame, currSRER, nMean
        
        # qhm iteration
        print("QHM")
        hFreqList = np.zeros((nHop, self.maxHar))
        hAmpList = np.zeros((nHop, self.maxHar))
        hPhaseList = np.zeros((nHop, self.maxHar))
        
        bestSRERList = np.full(nHop, -np.inf)
        for iHop, f0 in enumerate(self.f0List):
            if(f0 <= 0.0):
                continue
            print(iHop, "/", nHop)
            windowSize = int(np.ceil(np.ceil(self.samprate / f0) * B * 2.0))
            if(windowSize % 2 == 0):
                windowSize += 1
            aWindow = windowFunc(windowSize)
            iCenter = self.hopSize * iHop
            nHar = int(self.mvf / f0)
            cFrame = getFrame(x, iCenter, self.hopSize * 2) * sWindow # comparison frame
            for iIter in range(self.mainCorrIter + self.extraCorrIter):
                aFrame = getFrame(x, iCenter, windowSize) # analysis frame
                if(iIter == 0):
                    ak, bk, fk_hat, status = self.slove(aFrame, aWindow, f0, self.mvf, self.samprate, nIter = 0, maxCorr = min(20.0, 0.12 * f0))
                else:
                    ak, bk, fk_hat, status = self.slove(aFrame, aWindow, f0, self.mvf, self.samprate, nIter = 1, maxCorr = min(20.0, 0.12 * f0), doIterOn = status)
                sFrame = self.synthFromAK(ak, fk_hat, sRange, self.samprate) * sWindow
                currSRER = self.calcSRER(cFrame, sFrame)
                if(currSRER - bestSRERList[iHop] > 0.0):
                    bestSRERList[iHop] = currSRER
                    hFreqList[iHop], hAmpList[iHop], hPhaseList[iHop] = self.parameterFromAK(ak, fk_hat, self.maxHar)
                if(currSRER >= self.mainTargetSRER and iIter == self.mainCorrIter - 1):
                    break
        return self.f0List, hFreqList, hAmpList, hPhaseList, bestSRERList
        
    @staticmethod
    def slove(x, window, f0, mvf, sr, nIter = 0, maxCorr = 20.0, maxNHar = None, fk = None, lastFreqCorr = False, doIterOn = None):
        nyq = sr
        nX = len(x)
        if(len(window) != nX):
            raise ValueError("bad window length.")
        x = x.reshape(nX, 1)
        window = window.reshape(nX, 1)
        
        if(mvf < 4e3 or mvf >= nyq):
            raise ValueError("mvf must in range [4e3, nyq).")
        if(nX % 2 == 0):
            raise ValueError("length of x must be odd.")
        if(f0 <= 0):
            raise ValueError("bad f0. (it's a unvoiced frame?)")
        nHar = int(mvf / f0) if(maxNHar is None) else min(maxNHar, int(mvf / f0))
        # build
        if(doIterOn is None):
            K = nHar * 2 + 1 # num of har
            N = int((nX - 1) / 2) # half frame len (odd required).
            if(fk is None):
                fk_hat = np.arange(-nHar, nHar + 1).reshape(K, 1) * f0 # freq of har
            else:
                fk_hat = np.concatenate((-fk[::-1], (0,), fk)).reshape(K, 1)
            n = np.arange(-N, N + 1).reshape(1, nX) # time vec
            
            # LS Analysis
            t = (np.dot(2 * np.pi * fk_hat, n) / sr).T # arg of cplx exp
            E = np.cos(t) + 1j * np.sin(t) # mat with cplx exp, (2N + 1, K)
            E = np.concatenate((E, np.tile(n.T, (1, K)) * E), axis = 1) # (2N + 1, 2K)
            Ew = np.tile(window, (1, 2 * K)) * E # multiply the window
            R = np.dot(Ew.T, Ew) # compute the matrix to be inverted
            try:
                theta = np.dot(sla.pinv2(R), np.dot(Ew.T, x * window))
            except np.linalg.linalg.LinAlgError: # if SVD did not converge
                theta = sla.lstsq(R, np.dot(Ew.T, x * window))[0]
            ak = theta[:K] # cplx amps
            bk = theta[K:] # cplx slopes
        else:
            K, N, fk_hat, n, t, E, Ew, R, theta, ak, bk = doIterOn
        
        # iteration
        for iIter in range(nIter):
            fk_hat += np.clip(sr / (2 * np.pi) * (ak.real * bk.imag - ak.imag * bk.real) / (np.abs(ak) ** 2), -maxCorr, maxCorr)
            # LS Analysis
            t = (np.dot(2 * np.pi * fk_hat, n) / sr).T # arg of cplx exp
            E = np.cos(t) + 1j * np.sin(t) # mat with cplx exp, [2N + 1 * K] dim
            E = np.concatenate((E, np.tile(n.T, (1, K)) * E), axis = 1) # (2N + 1, 2K)
            Ew = np.tile(window, (1, 2 * K)) * E # multiply the window
            R = np.dot(Ew.T, Ew) # compute the matrix to be inverted
            try:
                theta = np.dot(sla.pinv2(R), np.dot(Ew.T, x * window))
            except np.linalg.linalg.LinAlgError: # if SVD did not converge
                theta = sla.lstsq(R, np.dot(Ew.T, x * window))[0]
            ak = theta[:K] # cplx amps
            bk = theta[K:] # cplx slopes
        if(lastFreqCorr):
            fk_hat += np.clip(sr / (2 * np.pi) * (ak.real * bk.imag - ak.imag * bk.real) / (np.abs(ak) ** 2), -maxCorr, maxCorr)
        return ak, bk, fk_hat, (K, N, fk_hat, n, t, E, Ew, R, theta, ak, bk)
    
    @staticmethod
    def calcSRER(x, y):
        return np.log10(np.std(x) / np.std(x - y)) * 20.0
    
    @staticmethod
    def synthFromAK(ak, fk_hat, r, sr):
        r = r.reshape(1, len(r))
        return np.dot(ak.T, np.exp(np.dot(2j * np.pi * fk_hat, r) / sr)).real
    
    @staticmethod
    def parameterFromAK(ak, fk_hat, maxNHar = None):
        nHar = int((len(fk_hat) - 1) / 2)
        if(maxNHar is None):
            maxNHar = nHar
        pNHar = min(nHar, maxNHar)
        hFreq = np.zeros(maxNHar)
        hAmp = np.zeros(maxNHar)
        hPhase = np.zeros(maxNHar)
        
        hFreq[:pNHar] = fk_hat.reshape(nHar * 2 + 1)[nHar + 1:nHar + 1 + pNHar]
        hAmp[:pNHar] = np.log10(np.abs(ak.reshape(nHar * 2 + 1)[nHar + 1:nHar + 1 + pNHar]) * 2.0) * 20.0
        hPhase[:pNHar] = np.unwrap(np.angle(ak.reshape(nHar * 2 + 1)[nHar + 1:nHar + 1 + pNHar]))
        return hFreq, hAmp, hPhase

def __test__():
    stft = __import__("stft")
    pyin = __import__("pyin")
    sr, w = loadWav("iA3.wav")
    yin = pyin.Model(sr)
    yin.bias = 2.0
    yin(w)
    
    qhm = QHM_AIR(yin.f0, sr)
    f0List, hFreqList, hAmpList, bestSRERList = qhm(w)
    print(bestSRERList)
    pl.figure()
    pl.plot(yin.f0)
    pl.plot(f0List)
    pl.figure()
    hFreqList[hFreqList <= 0.0] = np.nan
    pl.plot(hFreqList)
    pl.show()

if(__name__ == "__main__"):
    profile = __import__("profile")
    profile.run("__test__()")