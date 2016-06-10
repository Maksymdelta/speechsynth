import numpy as np
import scipy.signal as sp
import scipy.interpolate as ipl
import pylab as pl
import fecsola
import tune
import hnm
from common import *

class Processor:
    def __init__(self, voiceDB):
        self.voiceDB = voiceDB

    def __call__(self, proj, trk):
        if(trk.hasOverlapped()):
            raise ValueError('Note cannot be overlaped.')
        if(not trk.noteList):
            return np.zeros(0, dtype = np.float64)

        nyq = self.voiceDB.samprate / 2
        kernelSize = 51

        # pass0: gen f0
        length = proj.tickToTime(trk.endPos)
        nHop = int(np.ceil(length * self.voiceDB.samprate / self.voiceDB.hopSize))
        f0List = np.zeros(nHop + kernelSize)
        phoneList = []
        prevNoteEnd = 0
        for iNote, (pos, note) in enumerate(trk.noteList):
            if(prevNoteEnd != pos):
                begin = int(round(proj.tickToTime(prevNoteEnd) * self.voiceDB.samprate / self.voiceDB.hopSize))
                end = int(round(proj.tickToTime(pos) * self.voiceDB.samprate / self.voiceDB.hopSize))
                f0List[begin:end] = tune.pitchToFreq(note.key - 1)
            begin = int(round(proj.tickToTime(pos) * self.voiceDB.samprate / self.voiceDB.hopSize))
            end = int(round(proj.tickToTime(pos + note.length) * self.voiceDB.samprate / self.voiceDB.hopSize))
            f0List[begin:end] = tune.pitchToFreq(note.key)
            prevNoteEnd = pos + note.length
        f0List[end:nHop + kernelSize] = tune.pitchToFreq(trk.noteList[-1][1].key - 1)
        
        mavg = sp.boxcar(kernelSize) / kernelSize
        f0List = np.convolve(f0List, mavg)[kernelSize // 2:-(kernelSize // 2) - kernelSize]

        # pass1: gen hnm parameter and smoothen f0
        hFreqList = np.zeros((nHop, self.voiceDB.nHar))
        hAmpList = np.zeros((nHop, self.voiceDB.nHar))
        hPhaseList = np.zeros((nHop, self.voiceDB.nHar))
        hOFList = np.zeros((nHop, self.voiceDB.nHar))
        formantList = np.zeros((nHop, self.voiceDB.maxFormant, 3))
        sinusoidEnvList = np.zeros((nHop, self.voiceDB.fftSize // 2 + 1))
        noiseEnvList = np.zeros((nHop, self.voiceDB.fftSize // 2 + 1))
        sinusoidEnergyList = np.zeros(nHop)
        noiseEnergyList = np.zeros(nHop)
        voicedList = np.zeros(nHop, dtype = np.bool)

        prevNoteEnd = -np.inf
        for iNote, (pos, note) in enumerate(trk.noteList):
            if(iNote + 1 >= len(trk.noteList)):
                nextNotePos = np.inf
            else:
                nextNotePos = trk.noteList[iNote + 1][0]
            prev = None if(iNote <= 0) else trk.noteList[iNote - 1][1]
            next = None if(iNote + 1 >= len(trk.noteList)) else trk.noteList[iNote + 1][1]
            curr = note
            phoneType, phone = self.selectPhone(None if(prev is None) else prev.decoedPhone, curr.decoedPhone, None if(next is None) else next.decoedPhone)
            phone = phone[0]

            begin = int(round(proj.tickToTime(pos) * self.voiceDB.samprate / self.voiceDB.hopSize))
            end = int(round(proj.tickToTime(pos + note.length) * self.voiceDB.samprate / self.voiceDB.hopSize))
            newHop = end - begin
            if(prevNoteEnd == pos):
                newRisePos = 0
            else:
                newRisePos = phone.risePos
            if(pos + note.length == nextNotePos):
                newFallPos = newHop
            else:
                newFallPos = newHop - (len(phone.hOFList) - phone.fallPos)
            if(newRisePos > newHop // 2):
                newRisePos = newHop // 2
            if(newFallPos <= newRisePos):
                newFallPos = newRisePos + 1
            if(phoneType == 'c' or phoneType == 'v' or phoneType == 'b'):
                newHOFList, newPhaseList, newFormantList, newSinusoidEnvList, newNoiseEnvList, newSinusoidEnergyList, newNoiseEnergyList, newVoicedList = self.timeScaleSingle(phone, newHop, newRisePos, newFallPos)
                # place
                nVHar = newHOFList.shape[1]
                nTHar = self.voiceDB.nHar
                nVFormant = newFormantList.shape[1]
                nTFormant = self.voiceDB.maxFormant
                for iHop in range(begin, end):
                    if(nVHar >= nTHar):
                        hOFList[iHop] = newHOFList[iHop - begin][:nTHar]
                        hPhaseList[iHop] = newPhaseList[iHop - begin][:nTHar]
                        formantList[iHop] = newFormantList[iHop - begin][:nTFormant]
                    else:
                        hOFList[iHop][:nVHar] = newHOFList[iHop - begin]
                        hPhaseList[iHop][:nVHar] = newPhaseList[iHop - begin]
                        formantList[iHop][:nVFormant] = newFormantList[iHop - begin]
                voicedList[begin:end] = newVoicedList
                sinusoidEnvList[begin:end] = newSinusoidEnvList
                noiseEnvList[begin:end] = newNoiseEnvList
                sinusoidEnergyList[begin:end] = newSinusoidEnergyList
                noiseEnergyList[begin:end] = newNoiseEnergyList
                prevNoteEnd = pos + note.length

        # pass2: smoothen formant
        pl.plot(formantList.transpose(2, 1, 0)[0].T)
        pl.show()

        smoothRadius = 20
        for iNote, (pos, note) in enumerate(trk.noteList):
            nextNote = None if(iNote + 1 >= len(trk.noteList)) else trk.noteList[iNote + 1][1]
            if(iNote != 0): # prefix
                prevNotePos, prevNote = trk.noteList[iNote - 1]
                prevNoteLength = prevNote.length
                prevNoteEnd = prevNotePos + prevNoteLength
                pprevNote = None if(iNote <= 1) else trk.noteList[iNote - 2][1]
                if(prevNoteEnd == pos):
                    prevSmoothLength = min(prevNoteLength // 2, smoothRadius)
                    currSmoothLength = min(note.length // 2, smoothRadius)
                    currPhoneType, currPhone = self.selectPhone(prevNote.decoedPhone, curr.decoedPhone, None if(nextNote is None) else nextNote.decoedPhone)
                    prevPhoneType, prevPhone = self.selectPhone(None if(pprevNote is None) else pprevNote.decoedPhone, prevNote.decoedPhone, curr.decoedPhone)
                    currPhone = currPhone[0]
                    prevPhone = prevPhone[0]
                    begin = int(round(proj.tickToTime(pos) * self.voiceDB.samprate / self.voiceDB.hopSize))
                    fecsolaProcessor = fecsola.Processor(self.voiceDB.samprate)
                    fecsolaProcessor.hfFreq = phone.hfFreq
                    fecsolaProcessor.hfBW = phone.hfBW
                    formantAnchorA = formantList[begin - 1].copy()
                    formantAnchorB = formantList[begin].copy()
                    envAnchorA = sinusoidEnvList[begin - 1].copy()
                    envAnchorB = sinusoidEnvList[begin].copy()
                    for iDeltaHop in range(-(smoothRadius - 1), smoothRadius):
                        iHop = begin + iDeltaHop
                        #ratio = (smoothRadius + iDeltaHop) / (smoothRadius * 2)
                        ratio = np.sin(iDeltaHop / smoothRadius * 0.5 * np.pi) / 2.0 + 0.5
                        print(iDeltaHop, ratio)
                        if(iDeltaHop < 0):
                            sinusoidEnvList[iHop] = fecsolaProcessor.mix(sinusoidEnvList[iHop], formantList[iHop], envAnchorB, formantAnchorB, ratio)
                            formantList[iHop] = lerp(formantList[iHop], formantAnchorB, ratio)
                        else:
                            sinusoidEnvList[iHop] = fecsolaProcessor.mix(envAnchorA, formantAnchorA, sinusoidEnvList[iHop], formantList[iHop], ratio)
                            formantList[iHop] = lerp(formantAnchorA, formantList[iHop], ratio)

        pl.plot(formantList.transpose(2, 1, 0)[0].T)
        pl.show()

        np.multiply(hOFList, f0List.reshape(len(f0List), 1), out = hFreqList)
        fRange = np.fft.rfftfreq(self.voiceDB.fftSize, 1.0 / self.voiceDB.samprate)
        kill = hFreqList >= nyq
        hFreqList[kill] = 0.0
        f0List[~voicedList] = 0.0
        for iHop, voiced in enumerate(voicedList):
            if(not voiced):
                continue
            need = np.logical_and(hFreqList[iHop] > 0.0, hFreqList[iHop] < nyq)
            hAmpList[iHop][need] = ipl.interp1d(fRange, sinusoidEnvList[iHop], kind = 'linear', bounds_error = False, fill_value = sinusoidEnvList[iHop][-1])(hFreqList[iHop][need])


        # pass3: hnm synthesis
        model = hnm.Model(self.voiceDB.samprate)
        model.fftSize = self.voiceDB.fftSize
        model.hopSize = self.voiceDB.hopSize
        #sinusoidEnergyList = np.convolve(sinusoidEnergyList, mavg)[kernelSize // 2:-(kernelSize // 2)]
        #noiseEnergyList = np.convolve(noiseEnergyList, mavg)[kernelSize // 2:-(kernelSize // 2)]
        pl.plot(sinusoidEnergyList)
        pl.plot(noiseEnergyList)
        pl.show()

        return model.synth(f0List, hFreqList, hAmpList, hPhaseList, noiseEnvList, sinusoidEnergyList, noiseEnergyList)

    def selectPhone(self, prev, curr, next):
        try:
            phoneList = self.voiceDB.getAttr("phone", curr)
            return phoneList.type, phoneList.phones[phoneList.maximumMatchedContext(prev, curr, next)[0]]
        except KeyError:
            raise ValueError('Phone "%s" is not found.' % (str(curr)))

    def timeScaleSingle(self, phone, newHop, newRisePos, newFallPos):
        oldHop = len(phone.hOFList)
        oldFac = np.arange(oldHop) / (oldHop - 1)
        newFac = np.arange(newHop) / (newHop - 1)

        if(newRisePos > 0):
            origKey = [0.0, phone.risePos / oldHop]
            newKey = [0.0, newRisePos / newHop]
        else:
            origKey = [phone.risePos / oldHop]
            newKey = [0.0]
        origKey.append(phone.fallPos / oldHop)
        if(newFallPos < newHop):
            origKey.append(1.0)
            newKey.append(newFallPos / newHop)
        newKey.append(1.0)
        scaler = ipl.PchipInterpolator(newKey, origKey)
        scaleFac = scaler(newFac)
        newHOFList = ipl.interp1d(oldFac, phone.hOFList, kind = 'linear', axis = 0, bounds_error = False, fill_value = phone.hOFList[-1])(scaleFac)
        newHPhaseList = ipl.interp1d(oldFac, phone.hPhaseList, kind = 'linear', axis = 0, bounds_error = False, fill_value = phone.hPhaseList[-1])(scaleFac)
        ff = phone.formantList[-1].shape[0] * phone.formantList[-1].shape[1]
        newFormantList = ipl.interp1d(oldFac, phone.formantList, kind = 'linear', axis = 0, bounds_error = False, fill_value = phone.formantList[-1].reshape(ff))(scaleFac)
        newNoiseEnvList = ipl.interp1d(oldFac, phone.noiseEnvList, kind = 'linear', axis = 0, bounds_error = False, fill_value = phone.noiseEnvList[-1])(scaleFac)
        newSinusoidEnergyList = ipl.interp1d(oldFac, phone.sinusoidEnergyList, kind = 'linear', bounds_error = False, fill_value = phone.sinusoidEnergyList[-1])(scaleFac)
        newNoiseEnergyList = ipl.interp1d(oldFac, phone.noiseEnergyList, kind = 'linear', bounds_error = False, fill_value = phone.noiseEnergyList[-1])(scaleFac)

        fecsolaProcessor = fecsola.Processor(self.voiceDB.samprate)
        fecsolaProcessor.hfFreq = phone.hfFreq
        fecsolaProcessor.hfBW = phone.hfBW
        newSinusoidEnvList = np.zeros((newHop, phone.sinusoidEnvList.shape[1]))
        voicedList = np.zeros(newHop, dtype = np.bool)
        for iHop in range(newHop):
            origHop = int(scaler(iHop / (newHop - 1)) * (oldHop - 1))
            if(not phone.voiced[origHop]):
                continue
            origHopRes = np.fmod(origHop, 1.0)
            origHopA = int(origHop)
            origHopB = min(oldHop - 1, origHopA + 1)
            voicedList[iHop] = True
            newSinusoidEnvList[iHop] = fecsolaProcessor.mix(phone.sinusoidEnvList[origHopA], phone.formantList[origHopA], phone.sinusoidEnvList[origHopB], phone.formantList[origHopB], origHopRes)
        return newHOFList, newHPhaseList, newFormantList, newSinusoidEnvList, newNoiseEnvList, newSinusoidEnergyList, newNoiseEnergyList, voicedList
