import numpy as np
import MonoPitch
import SparseHMM
import pylab as pl
from common import *
import numba as nb

class Model:
    def __init__(self, samprate):
        self.samprate = int(samprate)
        self.valleyThreshold = 0.5
        self.probThreshold = 0.02
        self.minFreq = 80.0
        self.maxFreq = 1000.0
        self.weightPrior = 5

        self.pdf = SparseHMM.normalized_pdf(1.7, SparseHMM.au_to_b(1.7, 0.2), 0, 1, 100)

        self.lowAmp = 0.1
        self.bias = 1.0
        self.hopSize = roundUpToPowerOf2(self.samprate * 0.005)
        self.windowSize = roundUpToPowerOf2(self.samprate * 0.025)

        self.f0 = None

    def __call__(self, input):
        hops = []
        nHop = getNFrame(len(input), self.hopSize)

        for iHop in range(nHop):
            hop = getFrame(input, iHop * self.hopSize, self.windowSize)
            hops.append(self.processHop(hop))
        mp = MonoPitch.Processor(minFreq = 61.375, nBPS = 5, nPitch = int(np.ceil(np.log2(self.maxFreq / 61.735) * 12)))

        #hops = np.array(hops)
        self.f0 = mp.process(hops)
        for iHop in range(nHop):
            if(self.f0[iHop] <= 0):
                continue
            frame = getFrame(input, iHop * self.hopSize, 2 * self.hopSize)
            if(np.max(frame) == np.min(frame)):
                self.f0[iHop] = 0.0

    def processHop(self, input):
        buffSize = int(len(input) / 2)

        buff = self.fastDifference(input, buffSize)
        buff = self.cumulativeDifference(buff)

        valleys = self.findValleys(buff, self.minFreq, self.maxFreq, self.samprate, threshold = self.valleyThreshold)
        nValley = len(valleys)
        mean = np.sum(input[:buffSize]) / buffSize
        input -= mean

        freqProb = []
        probTotal = 0.0
        for iValley, valley in enumerate(valleys):
            freq = self.samprate / parabolicInterpolation(buff, valley)[0]
            prob = 0.0
            v0 = 1 if(iValley == 0) else (buff[valleys[iValley - 1]] + 1e-10)
            v1 = 0 if(iValley == nValley - 1) else buff[valleys[iValley + 1]] + 1e-10
            k = np.arange(int(v1 * 100), int(v0 * 100), dtype = np.int)
            probFac = np.ones(int(v0 * 100) - int(v1 * 100))
            probFac[buff[valley] >= k / 100.0] = 0.01
            prob = np.sum(self.pdf[k] * probFac)
            prob = min(prob, 0.99)
            prob *= self.bias
            freqProb.append((freq, prob))
            probTotal += prob
        freqProb = np.array(freqProb, dtype = np.float64)

        # weight & renormalize
        newProbTotal = 0.0
        for iValley, valley in enumerate(valleys):
            if(buff[valley] < self.probThreshold):
                freqProb[iValley][1] *= self.weightPrior
            newProbTotal += freqProb[iValley][1]
        if(nValley > 0 and newProbTotal != 0.0):
            freqProb.T[1] *= probTotal / newProbTotal
        return freqProb

    @staticmethod
    def fastDifference(input, outSize):
        outSize = int(outSize)
        out = np.zeros((outSize), dtype = input.dtype)

        frameSize = outSize * 2

        # POWER TERM CALCULATION
        # ... for the power terms in equation (7) in the Yin paper

        powerTerms = np.zeros((outSize), dtype = np.float64)
        powerTerms[0] = np.sum(input[:outSize] ** 2)

        # now iteratively calculate all others (saves a few multiplications)
        for i in range(1, outSize):
            powerTerms[i] = powerTerms[i - 1] - (input[i - 1] ** 2) + input[i + outSize] * input [i + outSize]

        # YIN-STYLE ACF via FFT
        # 1. data
        transformedAudio = np.fft.rfft(input)

        # 2.half of the data, disguised as a convolution kernel
        kernel = np.zeros((frameSize), dtype = np.float64)
        kernel[:outSize] = input[:outSize][::-1]
        transformedKernel = np.fft.rfft(kernel)

        # 3. convolution
        yinStyleACF = transformedAudio * transformedKernel
        transformedAudio = np.fft.irfft(yinStyleACF)

        # CALCULATION OF difference function
        # according to (7) in the Yin paper
        out = powerTerms[0] + powerTerms[:outSize] - 2 * transformedAudio.real[outSize - 1:-1]
        return out

    @staticmethod
    @nb.jit(nb.float64[:](nb.float64[:]), cache=True)
    def cumulativeDifference(input):
        out = input.copy()
        out[0] = 1.0
        sum = 0.0

        for i in range(1, len(out)):
            sum += out[i]
            if(sum == 0):
                out[i] = 1
            else:
                out[i] *= i / sum
        return out

    @staticmethod
    def findValleys(x, minFreq, maxFreq, sr, threshold = 0.5, step = 0.01):
        ret = []
        begin = max(1, int(sr / maxFreq))
        end = min(len(x) - 1, int(np.ceil(sr / minFreq)))
        for i in range(begin, end):
            prev = x[i - 1]
            curr = x[i]
            next = x[i + 1]
            if(prev > curr and next > curr and curr < threshold):
                threshold = curr - step
                ret.append(i)
        return ret
