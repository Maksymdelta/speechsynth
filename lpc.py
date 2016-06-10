import numpy as np
import scipy.signal as sp
import scipy.linalg as sla
import pylab as pl
from common import *

class LPC:
    windows = {
        'boxcar': (sp.boxcar, 1.1),
        'hanning': (sp.hanning, 1.6),
        'blackman': (sp.blackman, 1.73)
    }

    def __init__(self, order, f0List, sr, window = 'blackman', resample = None):
        self.f0List = f0List
        self.emphasisFreq = 50.0
        self.samprate = sr
        self.resample = resample
        self.order = order
        self.window = window
        self.coeff = np.zeros(0)
        self.hopSize = roundUpToPowerOf2(self.samprate * 0.005)
        self.xms = np.zeros(0)

    def __call__(self, x, unvoiced = True):
        resampled = self.samprate if self.resample is None else self.resample
        nX = len(x)
        nHop = getNFrame(nX, self.hopSize)
        ratio = resampled / self.samprate
        assert(len(self.f0List) == nHop)

        if(self.samprate != resampled):
            x = sp.resample(x, int(round(nX * ratio)), window = 'hanning')
        x = preEmphasis(x, self.emphasisFreq, resampled)

        self.coeff = np.zeros((nHop, self.order))
        self.xms = np.zeros(nHop)
        windowFunc, B = self.windows[self.window]
        for iHop, f0 in enumerate(self.f0List):
            if(not unvoiced and f0 <= 0.0):
                continue
            windowSize = int(np.ceil(resampled / f0 * B * 2.0)) if(f0 > 0.0) else self.hopSize * 2
            if(windowSize % 2 == 0):
                windowSize += 1
            window = windowFunc(windowSize)
            iCenter = int(round(self.hopSize * iHop * ratio))
            frame = getFrame(x, iCenter, windowSize) * window
            if(np.sum(frame) == 0.0):
                continue
            self.coeff[iHop], self.xms[iHop] = self.procSingleFrame(frame)
        return self.coeff, self.xms

    def procSingleFrame(self, x):
        raise NotImplementedError("This LPC Object is not implemented.")

    def toFormant(self, minFreq = 100.0, maxFreq = None):
        resampled = self.samprate if self.resample is None else self.resample
        nyq = resampled / 2
        nHop = len(self.f0List)
        nMaxFormant = int(np.ceil(self.order / 2))

        if(maxFreq is None):
            maxFreq = nyq - 100.0
        if(maxFreq <= minFreq or minFreq <= 0.0):
            raise ValueError("Invalid minFreq/maxFreq.")

        freqList = np.zeros((nHop, nMaxFormant))
        bwList = np.zeros((nHop, nMaxFormant))

        for iHop, f0 in enumerate(self.f0List):
            if(f0 <= 0.0):
                continue
            polyCoeff = np.zeros(self.order + 1)
            polyCoeff[:self.order] = self.coeff[iHop][::-1]
            polyCoeff[-1] = 1.0
            roots = np.roots(polyCoeff)
            roots = roots[roots.imag >= 0.0] # remove conjugate roots
            roots = fixIntoUnit(roots)

            freqs = []
            bws = []
            for iRoot in range(len(roots)):
                freq = np.abs(np.arctan2(roots.imag[iRoot], roots.real[iRoot])) * nyq / np.pi
                if(freq >= minFreq and freq <= maxFreq):
                    bw = -np.log(np.abs(roots[iRoot]) ** 2) * nyq / np.pi
                    freqs.append(freq)
                    bws.append(bw)

            freqs = np.array(freqs)
            bws = np.array(bws)

            sortOrder = np.argsort(freqs)
            nFormant = min(nMaxFormant, len(freqs))
            freqList[iHop][:nFormant] = freqs[sortOrder][:nFormant]
            bwList[iHop][:nFormant] = bws[sortOrder][:nFormant]

        return freqList, bwList

    def toSpectrum(self, nFFT, bandwidthReduction = 0.0):
        resampled = self.samprate if self.resample is None else self.resample
        nHop = len(self.coeff)

        out = np.full((nHop, int(nFFT / 2) + 1), 1e-10)
        for iHop, f0 in enumerate(self.f0List):
            out[iHop] = self.coeffToSpectrum(self.coeff[iHop], nFFT, bandwidthReduction, self.emphasisFreq, resampled, gain = self.xms[iHop])
        return np.log10(out) * 20.0

    @staticmethod
    def coeffToSpectrum(a, nfft, bandwidthReduction, deEmphasisFreq, sr, gain = 0.0):
        nyq = sr / 2
        ndata = len(a) + 1
        scale = 1 / np.sqrt(2 * nyq * nyq / (nfft / 2))

        # copy to buffer
        fftBuffer = np.zeros((nfft))
        fftBuffer[0] = 1.0
        fftBuffer[1:ndata] = a

        # deemphasis
        if(deEmphasisFreq > 0.0):
            fac = np.exp(-2 * np.pi * deEmphasisFreq / nyq)
            ndata += 1
            for i in reversed(range(1, ndata)):
                fftBuffer[i] -= fac * fftBuffer[i - 1]
        # reduce bandwidth
        if(bandwidthReduction > 0.0):
            # sum (k=0..ndata; a[k] (z)^-k)
            # sum (k=0..ndata; a[k] (rz)^-k) = sum (k=0..ndata; (a[k]r^-k) z^-k)
            fac = np.exp(np.pi * bandwidthReduction / sr)
            fftBuffer[1:ndata] *= np.power(fac, np.arange(2, ndata + 1))

        # do fft
        if(gain > 0.0):
            scale *= np.sqrt(gain)
        o = np.fft.rfft(fftBuffer)
        o.real[0] = scale / o.real[0]
        o.imag[0] = 0
        for i in range(1, int(nfft / 2)):
            o[i] = np.conj(o[i] * scale / (np.abs(o[i]) ** 2))
        o.real[-1] = scale / o.real[-1]
        o.imag[-1] = 0
        #o[-1] = o[-2]
        return np.abs(o)

class Burg(LPC):
    def __init__(self, order, f0List, sr, window = 'blackman', resample = None):
        super().__init__(order, f0List, sr, window = window, resample = resample)

    def procSingleFrame(self, x):
        n = len(x)
        m = self.order

        a = np.ones(m)
        aa = np.ones(m)
        b1 = np.ones(n)
        b2 = np.ones(n)

        # (3)
        xms = np.sum(x * x) / n

        if(xms <= 0):
            #raise ValueError("Empty/Zero input.")
            return np.zeros(m), 0.0

        # (9)
        b1[0] = x[0]
        b2[n - 2] = x[n - 1]
        b1[1:n - 1] = b2[:n - 2] = x[1:n - 1]

        for i in range(m):
            # (7)
            numer = np.sum(b1[:n - i - 1] * b2[:n - i - 1])
            deno = np.sum((b1[:n - i - 1] ** 2) + (b2[:n - i - 1] ** 2))
            if(deno <= 0):
                raise ValueError("Bad denominator(order is too large/x is too short?).")

            a[i] = 2.0 * numer / deno

            # (10)
            xms *= 1.0 - a[i] * a[i]

            # (5)
            a[:i] = aa[:i] - a[i] * aa[:i][::-1]

            if(i < m - 1):
                # (8)
                # NOTE: i -> i + 1
                aa[:i + 1] = a[:i + 1]

                for j in range(n - i - 2):
                    b1[j] -= aa[i] * b2[j]
                    b2[j] = b2[j + 1] - aa[i] * b1[j + 1]
        return -a, np.sqrt(xms * n)

class Autocorrelation(LPC):
    def __init__(self, order, f0List, sr, window = 'blackman', resample = None):
        super().__init__(order, f0List, sr, window = window, resample = resample)

    def procSingleFrame(self, x):
        return self.__fastSlove__(x, self.order)

    @staticmethod
    def __slowSlove__(x, m):
        n = len(x)
        p = m + 1
        r = np.zeros(p)
        nx = np.min((p, n))
        x = np.correlate(x, x, 'full')
        r[:nx] = x[n - 1:n+m]
        a = np.dot(sla.pinv2(sla.toeplitz(r[:-1])), -r[1:])
        gain = np.sqrt(r[0] + np.sum(a * r[1:]))
        return a, gain

    @staticmethod
    def __fastSlove__(x, m):
        n = len(x)
        # do autocorrelate via FFT
        nFFT = roundUpToPowerOf2(2 * n - 1)
        nx = np.min((m + 1, n))
        r = np.fft.irfft(np.abs(np.fft.rfft(x, n = nFFT) ** 2))
        r = r[:nx] / n
        a, e, k = Autocorrelation.levinson_1d(r, m)
        gain = np.sqrt(np.sum(a * r * n))

        return a[1:], gain

    @staticmethod
    def levinson_1d(r, order):
        r = np.atleast_1d(r)
        if r.ndim > 1:
            raise ValueError("Only rank 1 are supported for now.")
        n = r.size
        if n < 1:
            raise ValueError("Cannot operate on empty array !")
        elif order > n - 1:
            raise ValueError("Order should be <= size-1")
        if not np.isreal(r[0]):
            raise ValueError("First item of input must be real.")
        elif not np.isfinite(1/r[0]):
            raise ValueError("First item should be != 0")
        # Estimated coefficients
        a = np.empty(order + 1, r.dtype)
        # temporary array
        t = np.empty(order + 1, r.dtype)
        # Reflection coefficients
        k = np.empty(order, r.dtype)

        a[0] = 1.0
        e = r[0]

        for i in range(1, order+1):
            acc = r[i]
            for j in range(1, i):
                acc += a[j] * r[i-j]
            k[i-1] = -acc / e
            a[i] = k[i-1]
            for j in range(order):
                t[j] = a[j]
            for j in range(1, i):
                a[j] += k[i-1] * np.conj(t[i-j])
            e *= 1 - k[i-1] * np.conj(k[i-1])

        return a, e, k
