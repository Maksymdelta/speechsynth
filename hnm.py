import numpy as np
import scipy.signal as sp
import scipy.interpolate as ipl
import pylab as pl
from common import *
import stft
import qhm_slove
import lpc
import pickle
import envelope
import numba as nb

class Model:
    def __init__(self, sr):
        self.samprate = sr
        self.hopSize = roundUpToPowerOf2(self.samprate * 0.005)
        self.fftSize = roundUpToPowerOf2(self.samprate * 0.05)
        self.mvf = 20e3
        self.maxAIRIter = 10
        self.mainQHMIter = 8
        self.mainTargetSRER = 30.0
        self.extraQHMIter = 2
        self.method = 'qhm-air'

    def __call__(self, x, f0List, bakedSinusoid = None):
        f0List = f0List.copy()
        nX = len(x)
        nSpec = int(self.fftSize / 2) + 1
        nHop = getNFrame(nX, self.hopSize)
        nyq = self.samprate / 2
        assert(len(f0List) == nHop)

        x = x - np.mean(x) # dc adjustment

        maxF0 = np.max(f0List)
        maxHar = int(self.mvf / maxF0)
        stftAnalyzer = stft.AdaptiveAnalyzer(self.fftSize, self.hopSize, self.samprate, window = 'blackman')
        frameSize = self.hopSize * 2
        sWindow = sp.hanning(frameSize)
        sRange = np.arange(-self.hopSize, self.hopSize)
        if(not bakedSinusoid is None):
            f0List, hFreqList, hAmpList, hPhaseList, SRERList = bakedSinusoid
        else:
            if(maxF0 > 0.0):
                # stft analysis & f0 refinement
                magnList, phaseList, f0List = stftAnalyzer(x, f0List, refineF0 = True)

                if(self.method == 'qhm-air'):
                    # qhm analysis
                    qhm = qhm_slove.QHM_AIR(f0List, self.samprate)
                    qhm.maxHar = maxHar
                    qhm.maxAIRIter = self.maxAIRIter
                    qhm.mainCorrIter = self.mainQHMIter
                    qhm.mainTargetSRER = self.mainTargetSRER
                    qhm.extraCorrIter = self.extraQHMIter
                    f0List, hFreqList, hAmpList, hPhaseList, SRERList = qhm(x)
                elif(self.method == 'qfft'):
                    hFreqList, hAmpList, hPhaseList = np.zeros((nHop, maxHar)), np.zeros((nHop, maxHar)), np.zeros((nHop, maxHar))
                    SRERList = None
                    for iHop, f0 in enumerate(f0List):
                        f0List[iHop], hFreqList[iHop], hAmpList[iHop], hPhaseList[iHop] = self.findHarmonic(f0, magnList[iHop], phaseList[iHop], self.mvf, maxHar, self.samprate)
            else:
                hFreqList, hAmpList, hPhaseList = np.zeros((nHop, maxHar)), np.zeros((nHop, maxHar)), np.zeros((nHop, maxHar))

        # sort harmonic & resynth and & sinusoid energy
        sinusoid = np.zeros(len(x))
        sinusoidEnergyList = np.zeros(nHop)
        for iHop, f0 in enumerate(f0List):
            if(f0 <= 0.0):
                continue
            need = hFreqList[iHop] > 0.0
            order = np.argsort(hFreqList[iHop][need])
            hFreqList[iHop][need] = hFreqList[iHop][need][order]
            hAmpList[iHop][need] = hAmpList[iHop][need][order]
            hPhaseList[iHop][need] = hPhaseList[iHop][need][order]
            ob, oe, ib, ie = getFrameRange(nX, iHop * self.hopSize, frameSize)
            resynthed = self.synthSinusoid(hFreqList[iHop], hAmpList[iHop], hPhaseList[iHop], sRange, self.samprate) * sWindow
            sinusoid[ib:ie] += resynthed[ob:oe]
            need = hFreqList[iHop] > 0.0
            sinusoidEnergyList[iHop] = np.sqrt(np.sum(np.power(10, hAmpList[iHop][need] / 20.0) ** 2))

        if(bakedSinusoid is None):
            pickle.dump((f0List, hFreqList, hAmpList, hPhaseList, SRERList), open("sin.pickle", "wb"))

        noise = x - sinusoid

        # build noise envelope
        stftAnalyzer.window = "hanning"
        magnList, phaseList, noiseEnergyList = stftAnalyzer(noise, f0List, energy = True)
        envGen = envelope.MFIEnvelope(self.samprate)
        noiseEnv = envGen(noise, f0List)
        for iHop, f0 in enumerate(f0List):
            if(noiseEnergyList[iHop] != 0.0):
                noiseEnv[iHop] = np.log10(np.power(10, noiseEnv[iHop] / 20.0) / noiseEnergyList[iHop]) * 20.0

        # sync phase
        need = f0List > 0.0
        f0Need = f0List[need]
        f0Need = f0Need.reshape((len(f0Need), 1))
        base = hPhaseList[need].T[0].reshape((len(f0Need), 1))
        hPhaseList[need] -= base * (hFreqList[need] / f0Need)
        hPhaseList[need] = np.mod(np.mod(hPhaseList[need], 2 * np.pi) + 3.0 * np.pi, 2 * np.pi) - np.pi

        # resynth
        sinPS = np.zeros(len(x))
        for iHop, f0 in enumerate(f0List):
            if(f0 <= 0.0):
                continue
            ob, oe, ib, ie = getFrameRange(nX, iHop * self.hopSize, frameSize)
            sinPS[ib:ie] += (self.synthSinusoid(hFreqList[iHop], hAmpList[iHop], hPhaseList[iHop], sRange, self.samprate) * sWindow)[ob:oe]
        saveWav("sin.wav", sinusoid, self.samprate)
        saveWav("noise.wav", noise, self.samprate)
        saveWav("sinPS.wav", sinPS, self.samprate)
        return f0List, hFreqList, hAmpList, hPhaseList, sinusoidEnergyList, noiseEnergyList, noiseEnv

    def synth(self, f0List, hFreqList, hAmpList, hPhaseList, noiseEnv, sinusoidEnergyList, noiseEnergyList, sinusoidOn = True, noiseOn = True):
        nHop = len(f0List)
        nOut = nHop * self.hopSize
        nHar = hFreqList.shape[1]
        olaFac = 2
        frameSize = self.hopSize * 2
        sWindow = sp.hanning(frameSize)
        sRange = np.arange(-self.hopSize, self.hopSize)
        nyq = self.samprate / 2

        # sync phase
        syncBase = np.zeros(nHop * olaFac)
        syncedHPhaseList = np.zeros((nHop * olaFac, nHar))
        basePhase = 0.0
        for iFrame in range(1, nHop * olaFac):
            iHop = int(iFrame / olaFac)
            f0 = f0List[iHop]
            if(f0 <= 0.0):
                continue
            basePhase += f0 * self.hopSize / olaFac / self.samprate * 2 * np.pi
            syncBase[iFrame] = np.mod(basePhase, 2 * np.pi) - np.pi
            syncedHPhaseList[iFrame] = syncBase[iFrame] / f0 * hFreqList[iHop] + hPhaseList[iHop]
        # sinusoid
        sinusoid = np.zeros(nOut)
        for iFrame in range(nHop * olaFac):
            iHop = int(iFrame / olaFac)
            f0 = f0List[iHop]
            if(f0 <= 0.0):
                continue
            iCenter = int(iFrame * self.hopSize / olaFac)
            ob, oe, ib, ie = getFrameRange(nOut, iCenter, frameSize)
            need = hFreqList[iHop] > 0.0
            lamp = np.power(10, hAmpList[iHop] / 20.0)
            amp = np.log10(lamp * (sinusoidEnergyList[iHop] / np.sqrt(np.sum(lamp[need] ** 2)))) * 20.0
            frame = self.synthSinusoid(hFreqList[iHop], amp, syncedHPhaseList[iFrame], sRange, self.samprate) * sWindow
            sinusoid[ib:ie] += frame[ob:oe]
        sinusoid /= olaFac
        saveWav("sinrs.wav", sinusoid, self.samprate)

        if(noiseOn):
            # noise
            noiseTemplate = np.random.uniform(-1.0, 1.0, nOut)
            noise = np.zeros(nOut)
            noiseEnv = noiseEnv.copy()
            eSinusoid = sinusoid - self.lowerEnv(sinusoid, f0List, self.hopSize, self.samprate)
            eSinusoid[eSinusoid < 0.0] = 0.0
            for iHop in range(nHop):
                if(noiseEnergyList[iHop] <= 0.0):
                    continue
                noiseEnv[iHop] = np.log10(np.power(10, noiseEnv[iHop] / 20.0) * noiseEnergyList[iHop]) * 20.0
            noiseTemplate = self.filterNoise(noiseTemplate, noiseEnv, self.hopSize)
            for iFrame in range(nHop * olaFac):
                # energy normalization
                iCenter = int(round(iFrame * self.hopSize / olaFac))
                iHop = int(iFrame / olaFac)
                f0 = f0List[iHop]
                ob, oe, ib, ie = getFrameRange(nOut, iCenter, frameSize)

                noiseFrame = getFrame(noiseTemplate, iCenter, frameSize)
                noiseFrame /= np.sqrt(np.mean(noiseFrame ** 2))

                if(f0 > 0.0):
                    sinusoidFrameSize = int(self.samprate / f0 * 2)
                    if(sinusoidFrameSize % 2 == 1):
                        sinusoidFrameSize += 1
                    sinusoidEnergyFrame = getFrame(eSinusoid, iCenter, sinusoidFrameSize)
                    sinusoidFrame = getFrame(eSinusoid, iCenter, frameSize)
                    sinusoidFrame /= np.sqrt(np.mean(sinusoidEnergyFrame ** 2))
                    noiseFrame *= sinusoidFrame

                noiseFrame *= noiseEnergyList[iHop]
                noiseFrame *= sWindow
                noise[ib:ie] += noiseFrame[ob:oe]
            noise /= olaFac
            _, w, = loadWav('noise.wav')
        out = np.zeros(nOut)
        if(sinusoidOn):
            saveWav("sinrs.wav", sinusoid, self.samprate)
            out += sinusoid
        if(noiseOn):
            saveWav("noisers.wav", noise, self.samprate)
            out += noise
        saveWav("resynth.wav", out, self.samprate)
        return out

    @staticmethod
    def filterNoise(x, noiseFilter, hopSize):
        olaFac = 2
        nHop, filterSize = noiseFilter.shape
        olaHopSize = int(hopSize / olaFac)
        windowSize = hopSize * 4
        specSize = int(windowSize / 2) + 1
        nX = len(x)

        window = sp.hanning(windowSize)
        aNormFac = 0.5 * np.sum(window)
        sNormFac = 0.0
        for i in range(0, windowSize, olaHopSize):
            sNormFac += window[i]

        window = np.sqrt(window)
        buff = np.zeros(specSize, dtype = np.complex128)
        out = np.zeros(nX)
        for iFrame in range(nHop * olaFac):
            iHop = int(iFrame / olaFac)
            iCenter = iFrame * olaHopSize
            frame = getFrame(x, iCenter, windowSize)
            if((frame == 0.0).all()):
                continue
            env = ipl.interp1d(np.arange(specSize, step = specSize / filterSize), noiseFilter[iHop], kind = "linear")(np.arange(specSize))
            ffted = np.fft.rfft(frame * window)
            magn = np.abs(ffted)
            magn = np.power(10, env / 20.0) * aNormFac
            phase = np.angle(ffted)
            buff.real = magn * np.cos(phase)
            buff.imag = magn * np.sin(phase)
            o = np.fft.irfft(buff)
            o *= window
            ob, oe, ib, ie = getFrameRange(nX, iCenter, windowSize)
            out[ib:ie] += o[ob:oe] / sNormFac
        return out

    @staticmethod
    @nb.jit(nb.types.Tuple((nb.float64, nb.float64[:], nb.float64[:], nb.float64[:]))(nb.float64, nb.float64[:], nb.float64[:], nb.float64, nb.int32, nb.float64, nb.int32, nb.float64), cache=True)
    def findHarmonic(f0, magn, phase, mvf, nHar, sr, nAvgHar = 5, maxOffsetFac = 0.3):
        nSpec = len(magn)
        nAvgHar = min(nAvgHar, nHar)
        fftSize = (nSpec - 1) * 2
        outFreq = np.zeros((nHar))
        outAmp = np.zeros((nHar))
        outPhase = np.zeros((nHar))
        maxOffset = maxOffsetFac * f0
        for iHar in range(1, nAvgHar + 1):
            freq = iHar * f0
            if(freq >= mvf):
                break
            lowerIdx = int(max(0, np.floor((freq - maxOffset) / sr * fftSize)))
            upperIdx = int(min(nSpec - 1, np.ceil((freq + maxOffset) / sr * fftSize)))
            peakBin = Model.findBestIdx(magn, lowerIdx, upperIdx, int(round(freq / sr * fftSize)))
            peakBin, peakAmp = parabolicInterpolation(magn, peakBin)
            outFreq[iHar - 1] = peakBin * sr / fftSize
            outAmp[iHar - 1] = peakAmp
            outPhase[iHar - 1] = lerp(phase[int(peakBin)], phase[int(peakBin) + 1], np.mod(peakBin, 1.0))
        f0 = np.mean(outFreq[:nAvgHar] / np.arange(1, nAvgHar + 1))
        for iHar in range(nAvgHar + 1, nHar + 1):
            freq = iHar * f0
            if(freq >= mvf):
                break
            lowerIdx = int(max(0, np.floor((freq - maxOffset) / sr * fftSize)))
            upperIdx = int(min(nSpec - 1, np.ceil((freq + maxOffset) / sr * fftSize)))
            peakBin = Model.findBestIdx(magn, lowerIdx, upperIdx, int(round(freq / sr * fftSize)))
            peakBin, peakAmp = parabolicInterpolation(magn, peakBin)
            outFreq[iHar - 1] = peakBin * sr / fftSize
            outAmp[iHar - 1] = peakAmp
            outPhase[iHar - 1] = lerp(phase[int(peakBin)], phase[int(peakBin) + 1], np.mod(peakBin, 1.0))

        return f0, outFreq, outAmp, outPhase

    @staticmethod
    def findBestIdx(magn, lowerIdx, upperIdx, estIdx):
        if(lowerIdx == upperIdx):
            return lowerIdx
        rcmp = magn[lowerIdx + 1:upperIdx - 1]
        iPeaks = np.arange(lowerIdx + 1, upperIdx - 1)[np.logical_and(np.greater(rcmp, magn[lowerIdx:upperIdx - 2]), np.greater(rcmp, magn[lowerIdx + 2:upperIdx]))]
        if(len(iPeaks) == 0):
            return lowerIdx + np.argmax(magn[lowerIdx:upperIdx])
        else:
            return iPeaks[np.argmax(magn[iPeaks])]

    @staticmethod
    def synthSinusoid(hFreq, hAmp, hPhase, r, sr):
        out = np.zeros(len(r))
        nHar = len(hFreq)
        nyq = sr / 2

        hAmp = np.power(10, hAmp / 20.0)

        for iHar in range(nHar):
            freq = hFreq[iHar]
            amp = hAmp[iHar]
            phase = hPhase[iHar]
            if(freq <= 0.0 or freq >= nyq):
                break
            freqPhase = 2.0 * np.pi / sr * freq
            out[:] += np.cos(freqPhase * r + phase) * amp
        return out

    @staticmethod
    def lowerEnv(x, f0List, hopSize, samprate):
        nX = len(x)
        nHop = len(f0List)

        inputTime = nX / samprate
        instantList = []
        windowSizeList = []
        t = 0.0
        while(t < inputTime):
            iHop = int(t * samprate / hopSize)
            cf0 = max(100, f0List[min(iHop, nHop - 1)])
            windowSize = int(samprate / cf0)
            if(windowSize % 2 == 1):
                windowSize -= 1
            instantList.append(int(round(t * samprate)))
            windowSizeList.append(windowSize)
            t += 1.0 / cf0

        valIdxList = np.zeros(len(instantList), dtype = np.int)
        for iInstant, instant in enumerate(instantList):
            windowSize = windowSizeList[iInstant]
            frame = getFrame(x, instant, windowSize)
            iMin = np.argmin(frame)
            iMax = np.argmax(frame)
            if(frame[iMin] == frame[iMax]):
                valIdxList[iInstant] = max(0, min(nX, instant))
            else:
                valIdxList[iInstant] = max(0, min(nX, instant - int(windowSize / 2) + iMin))

        valList = x[valIdxList]
        out = ipl.interp1d(instantList, valList, kind = 'linear', bounds_error = False, fill_value = 0.0)(np.arange(nX))
        return out
