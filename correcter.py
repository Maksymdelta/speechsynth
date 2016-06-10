import numpy as np
import scipy.signal as sp
import scipy.interpolate as ipl
import pylab as pl
from common import *

class Processor:
    def __init__(self, sr):
        self.samprate = sr
        self.nIterF = 3
        self.nIterC = 13
        self.responseRadius = 1000.0
        self.preEmphasisFreq = 50.0
        self.integralSample = 128
        self.minimumFreq = 100.0
        self.maximumFreq = 6000.0
        self.minimumExcitationAmp = np.power(10.0, -120.0 / 20.0)

    def sloveExcitation(self, hFreq, hAmp, formant, rate = 1.0):
        nHar = len(hFreq)
        nFormant = len(formant)
        nyq = self.samprate / 2

        estAmp = 0.0
        for iFormant, (F, bw, amp) in enumerate(formant):
            if(iFormant == 0): # skip excitation
                continue
            estAmp += klattFilter(hFreq[0], amp, F, bw, self.samprate)

        eF, eBw, _ = formant[0]
        deltaAmp = max(self.minimumExcitationAmp, hAmp[0] - estAmp)
        unitAmp = klattFilter(hFreq[0], 1.0, eF, eBw, self.samprate)
        corrAmp = deltaAmp / unitAmp
        return corrAmp * rate

    def correctF(self, hFreq, hAmp, formant, rate = 1.0, dbg = None): # linear amp
        formant = formant.copy()
        nHar = len(hFreq)
        nFormant = len(formant)
        nyq = self.samprate / 2

        estAmp = np.zeros(nHar)
        for iFormant, (F, bw, amp) in enumerate(formant):
            estAmp += klattFilter(hFreq, amp, F, bw, self.samprate)
        deltaAmp = estAmp - hAmp
        integralRatio = np.arange(0, self.integralSample) / self.integralSample
        for iFormant, (F, bw, amp) in enumerate(formant):
            if(iFormant == 0): # skip excitation
                continue
            lowerFreq = max(0.0, F - self.responseRadius)
            if(iFormant == nFormant - 1):
                upperFreq = min(nyq, F + min(self.responseRadius, bw))
            else:
                upperFreq = min(nyq, F + self.responseRadius)
            lowerRange = F + (lowerFreq - F) * integralRatio
            higherRange = F + (upperFreq - F) * integralRatio
            # generate integration sample point
            iplY = np.concatenate(((deltaAmp[0], ), deltaAmp, (deltaAmp[-1], )))
            iplX = np.concatenate(((0.0, ), hFreq, (nyq, )))
            sampler = ipl.interp1d(iplX, iplY, kind='linear')
            leftMean = np.mean(sampler(lowerRange))
            rightMean = np.mean(sampler(higherRange))
            sym = -1 if((leftMean - rightMean) < 0) else 1
            formant[iFormant][0] = min(self.maximumFreq, max(self.minimumFreq, F + sym * np.sqrt(np.abs(leftMean - rightMean)) * (300.0 * rate)))
        if(dbg):
            pl.plot(hFreq, np.log10(hAmp) * 20.0)
            pl.plot(hFreq, np.log10(estAmp) * 20.0)
            estAmp = np.zeros(nHar)
            for iFormant, (F, bw, amp) in enumerate(formant):
                estAmp += klattFilter(hFreq, amp, F, bw, self.samprate)
            pl.plot(hFreq, np.log10(estAmp) * 20.0)
            pl.show()

        order = np.argsort(formant.T[0])
        formant = formant[order]
        return formant

    def correctAmp(self, hFreq, hAmp, formant, rate = 1.0, dbg = None): # linear amp
        formant = formant.copy()
        nHar = len(hFreq)
        nFormant = len(formant)
        nyq = self.samprate / 2

        estAmp = np.zeros(nHar)
        for iFormant, (F, bw, amp) in enumerate(formant):
            estAmp += klattFilter(hFreq, amp, F, bw, self.samprate)
        deltaAmp = estAmp - hAmp
        integralRatio = np.arange(0, self.integralSample) / self.integralSample
        for iFormant, (F, bw, amp) in enumerate(formant):
            if(iFormant == 0): # skip excitation
                continue
            lowerFreq = max(0.0, F - bw * 2)
            if(iFormant == nFormant - 1):
                upperFreq = min(nyq, F + bw)
            else:
                upperFreq = min(nyq, F + bw * 2)
            range = lowerFreq + (upperFreq - lowerFreq) * integralRatio
            # generate integration sample point
            iplY = np.concatenate(((deltaAmp[0], ), deltaAmp, (deltaAmp[-1], )))
            iplX = np.concatenate(((0.0, ), hFreq, (nyq, )))
            sampler = ipl.interp1d(iplX, iplY, kind='linear')
            mean = np.mean(sampler(range)) # TODO: add to realamp
            sym = 1 if(mean < 0) else -1
            #print("Amp:", amp + sym * np.abs(mean), min(1.0, max(1e-10, amp + sym * np.abs(mean) * rate * 2)), "Delta:", sym * np.abs(mean) * rate * 2)
            formant[iFormant][2] = min(1.0, max(1e-10, amp + sym * np.abs(mean) * rate * 2))
        if(dbg):
            pl.plot(hFreq, np.log10(hAmp) * 20.0)
            pl.plot(hFreq, np.log10(estAmp) * 20.0)
            estAmp = np.zeros(nHar)
            for iFormant, (F, bw, amp) in enumerate(formant):
                estAmp += klattFilter(hFreq, amp, F, bw, self.samprate)
            pl.plot(hFreq, np.log10(estAmp) * 20.0)
            pl.show()
        return formant

    def correctBW(self, hFreq, hAmp, formant, rate = 1.0, dbg = None): # linear amp
        formant = formant.copy()
        nHar = len(hFreq)
        nFormant = len(formant)
        nyq = self.samprate / 2

        estAmp = np.zeros(nHar)
        for iFormant, (F, bw, amp) in enumerate(formant):
            estAmp += klattFilter(hFreq, amp, F, bw, self.samprate)
        deltaAmp = estAmp - hAmp
        integralRatio = np.arange(0, self.integralSample) / self.integralSample
        for iFormant, (F, bw, amp) in enumerate(formant):
            if(iFormant == 0): # skip excitation
                continue
            lowerFreq = max(0.0, F - bw * 4)
            if(iFormant == nFormant - 1):
                upperFreq = min(nyq, F + bw * 0.05)
            else:
                upperFreq = min(nyq, F + bw * 4)
            range = lowerFreq + (upperFreq - lowerFreq) * integralRatio
            # generate integration sample point
            iplY = np.concatenate(((deltaAmp[0], ), deltaAmp, (deltaAmp[-1], )))
            iplX = np.concatenate(((0.0, ), hFreq, (nyq, )))
            sampler = ipl.interp1d(iplX, iplY, kind='linear')
            mean = np.mean(sampler(range)) # TODO: add to realamp
            sym = 1 if(mean < 0) else -1
            #print("BW:", bw + sym * np.sqrt(np.abs(mean)) * 1000.0 * rate, "Delta:", sym * np.sqrt(np.abs(mean)) * 1000 * rate)
            formant[iFormant][1] = min(800.0, max(80.0, bw + sym * np.sqrt(np.abs(mean)) * 1000.0 * rate))
        if(dbg):
            pl.plot(hFreq, np.log10(hAmp) * 20.0)
            pl.plot(hFreq, np.log10(estAmp) * 20.0)
            estAmp = np.zeros(nHar)
            for iFormant, (F, bw, amp) in enumerate(formant):
                estAmp += klattFilter(hFreq, amp, F, bw, self.samprate)
            pl.plot(hFreq, np.log10(estAmp) * 20.0)
            pl.show()
        return formant

    def correct(self, hFreq, hAmp, formant, dbg = None):
        order = np.argsort(formant.T[0])
        formant = formant[order]

        # amp to linear space
        hAmp = np.power(10, hAmp / 20.0) * preEmphasisResponse(hFreq, self.preEmphasisFreq, self.samprate)
        formant.T[2] = np.power(10, formant.T[2] / 20.0)
        rFormant = formant.copy()

        # correct F
        for iIter in range(self.nIterF):
            formant[0][2] = self.sloveExcitation(hFreq, hAmp, formant, rate = 0.98)
            formant = self.correctF(hFreq, hAmp, formant)

        if((np.diff(formant.T[0]) < 300.0).any()):
            formant = rFormant
        # correct F, bw, amp
        for iIter in range(self.nIterC):
            formant[0][2] = self.sloveExcitation(hFreq, hAmp, formant, rate = 0.98)
            formant = self.correctAmp(hFreq, hAmp, formant)
            formant = self.correctBW(hFreq, hAmp, formant, dbg = dbg)
            formant = self.correctF(hFreq, hAmp, formant)
        formant[0][2] = self.sloveExcitation(hFreq, hAmp, formant, rate = 0.98)
        formant = self.correctAmp(hFreq, hAmp, formant)
        formant[0][2] = self.sloveExcitation(hFreq, hAmp, formant, rate = 1.0)

        formant.T[2] = np.log10(formant.T[2]) * 20.0
        return formant

    def ref(self, hFreqList, hAmpList, envList, iRefHop, refFormant, trustAmp = False):
        nHop = len(hFreqList)
        nSpec = envList.shape[1]
        nFFT = (nSpec - 1) * 2
        fRange = np.arange(nSpec) / nFFT * self.samprate
        pe = preEmphasisResponse(fRange, self.preEmphasisFreq, self.samprate)

        correctedFormant = np.zeros((nHop, refFormant.shape[0], 3))
        # bilateral correct
        need = hFreqList[iRefHop] > 0.0
        if(not need.all()):
            raise ValueError("Bad refHop")
        correctedFormant[iRefHop] = self.correct(hFreqList[iRefHop][need], hAmpList[iRefHop][need], refFormant)

        for iHop in reversed(range(iRefHop)):
            need = hFreqList[iHop] > 0.0
            if(not need.any()):
                continue
            print(iHop)
            reference = correctedFormant[iHop + 1]
            if(not trustAmp):
                sampler = ipl.interp1d(fRange, np.log10(np.power(10, envList[iHop] / 20.0) * pe) * 20.0, kind='linear')
                reference[1:].T[2] = sampler(reference[1:].T[0]) + 3.0 # skip excitation
            correctedFormant[iHop] = self.correct(hFreqList[iHop][need], hAmpList[iHop][need], reference)#, dbg = True if(iHop == 60) else False)
        for iHop in range(iRefHop + 1, nHop):
            need = hFreqList[iHop] > 0.0
            if(not need.any()):
                continue
            print(iHop)
            reference = correctedFormant[iHop - 1]
            if(not trustAmp):
                sampler = ipl.interp1d(fRange, np.log10(np.power(10, envList[iHop] / 20.0) * pe) * 20.0, kind='linear')
                reference[1:].T[2] = sampler(reference[1:].T[0]) + 3.0 # skip excitation
            correctedFormant[iHop] = self.correct(hFreqList[iHop][need], hAmpList[iHop][need], reference)
        return correctedFormant

    def seq(self, hFreqList, hAmpList, envList, formantList, trustAmp = False):
        nHop = len(hFreqList)
        nSpec = envList.shape[1]
        nFFT = (nSpec - 1) * 2
        fRange = np.arange(nSpec) / nFFT * self.samprate
        pe = preEmphasisResponse(fRange, self.preEmphasisFreq, self.samprate)
        assert(formantList.shape[0] == nHop)

        correctedFormant = np.zeros((nHop, formantList.shape[1], 3))
        for iHop in range(nHop):
            need = hFreqList[iHop] > 0.0
            if(not need.any()):
                continue
            print(iHop)
            reference = formantList[iHop]
            if(not trustAmp):
                sampler = ipl.interp1d(fRange, np.log10(np.power(10, envList[iHop] / 20.0) * pe) * 20.0, kind='linear')
                reference[1:].T[2] = sampler(reference[1:].T[0]) + 3.0 # skip excitation
            correctedFormant[iHop] = self.correct(hFreqList[iHop][need], hAmpList[iHop][need], reference)

        return correctedFormant
