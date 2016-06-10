import numpy as np
import scipy.io.wavfile as wavfile
import scipy.interpolate as ipl
import numba as nb

def genSegments(data, dbg = False):
    segments = []
    if(data[0] > 0.0):
        segments.append(0)
    if(data.dtype == np.bool):
        for iHop in range(1, len(data)):
            if(data[iHop - 1] == False and data[iHop] == True):
                segments.append(iHop)
            elif(data[iHop] == False and data[iHop - 1] == True):
                segments.append(iHop)
    else:
        for iHop in range(1, len(data)):
            if(data[iHop - 1] <= 0 and data[iHop] > 0):
                segments.append(iHop)
            elif(data[iHop] <= 0 and data[iHop - 1] > 0):
                segments.append(iHop)
    if(len(segments) % 2):
        segments.append(len(data))
    segments = np.array(segments).reshape((len(segments) / 2, 2))
    return segments

def loadWav(filename): # -> samprate, wave in float64
    samprate, w = wavfile.read(filename)
    if(w.dtype == np.int8):
        w = w.astype(np.float64) / 128.0
    elif(w.dtype == np.short):
        w = w.astype(np.float64) / 32768.0
    elif(w.dtype == np.int32):
        w = w.astype(np.float64) / 2147483648.0
    elif(w.dtype == np.float32):
        w = w.astype(np.float64)
    elif(w.dtype == np.float64):
        pass
    else:
        raise ValueError("Unsupported sample format: %s" % (str(w.dtype)))
    return samprate, w

def saveWav(filename, data, samprate):
    wavfile.write(filename, samprate, data)

@nb.jit(nb.types.Tuple((nb.int64, nb.int64, nb.int64, nb.int64))(nb.int64, nb.int64, nb.int64), nopython=True, cache=True)
def getFrameRange(inputLen, center, size):
    leftSize = int(size / 2)
    rightSize = size - leftSize # for odd size

    inputBegin = min(inputLen, max(center - leftSize, 0))
    inputEnd = max(0, min(center + rightSize, inputLen))

    outBegin = max(leftSize - center, 0)
    outEnd = outBegin + (inputEnd - inputBegin)

    return outBegin, outEnd, inputBegin, inputEnd

@nb.jit(nb.float64[:](nb.float64[:], nb.int64, nb.int64), nopython=True, cache=True)
def getFrame(input, center, size):
    out = np.zeros((size), input.dtype)

    outBegin, outEnd, inputBegin, inputEnd = getFrameRange(len(input), center, size)

    out[outBegin:outEnd] = input[inputBegin:inputEnd]
    return out

@nb.jit(nb.int64(nb.int64, nb.int64), nopython=True, cache=True)
def getNFrame(inputSize, hopSize):
    return int(inputSize / hopSize + 1 if(inputSize % hopSize != 0) else inputSize / hopSize)

def roundUpToPowerOf2(v):
    return int(2 ** np.ceil(np.log2(v)))

def parabolicInterpolation(input, i, val = True, overAdjust = False):
    lin = len(input)

    ret = 0.0
    if(i > 0 and i < lin - 1):
        s0 = float(input[i - 1])
        s1 = float(input[i])
        s2 = float(input[i + 1])
        a = (s0 + s2) / 2.0 - s1
        if(a == 0):
            return (i, input[i])
        b = s2 - s1 - a
        adjustment = -(b / a * 0.5)
        if(not overAdjust and abs(adjustment) > 1.0):
            adjustment = 0.0
        x = i + adjustment
        if(val):
            y = a * adjustment * adjustment + b * adjustment + s1
            return (x, y)
        else:
            return x
    else:
        x = i
        if(val):
            y = input[x]
            return (x, y)
        else:
            return x

def lerp(v0, v1, ratio):
    return v0 + (v1 - v0) * ratio

def addPoint(l, x, minDist = 0):
    if(minDist > 0):
        bad = np.arange(len(l))[np.abs(np.asarray(l) - x) < minDist]
        for i in reversed(bad):
            l.pop(i)
    try:
        i = l.index(x)
        return i
    except ValueError:
        try:
            i = next(idx for idx, val in enumerate(l) if val > x)
            l.insert(i, x)
            return i
        except StopIteration:
            l.append(x)
            return len(l) - 1

def gaussianWindow(M):
    iMid = M * 0.5
    edge = np.exp(-12.0)
    phase = (np.arange(M) - iMid) / M
    return (np.exp(-48.0 * phase * phase) - edge) / (1.0 - edge)

def trueEnv(spec, order, iterCount = 32, maxStep = 1.5):
    # initialize the iteration using A0(k) = log(|X(k)|)
    a = spec.copy().astype(np.complex128)

    # prepare iter
    lastC = np.fft.irfft(a)
    lastC[order:-order] = 0.0
    v = np.fft.rfft(lastC)

    less = a.real < v.real
    a.real[less] = v.real[less]
    lastC = np.fft.irfft(a)
    lastC[order:-order] = 0.0
    v = np.fft.rfft(lastC)

    for iIter in range(iterCount):
        step = np.power(maxStep, (iterCount - iIter) / iterCount)
        less = a.real < v.real
        a.real[less] = v.real[less]
        c = np.fft.irfft(a)
        lastC[:order] = c[:order] + (c[:order] - lastC[:order]) * step
        lastC[-order:] = c[-order:] + (c[-order:] - lastC[-order:]) * step
        lastC[order:-order] = 0.0
        v = np.fft.rfft(lastC)
    return v.real

def fixIntoUnit(x):
    if(isinstance(x, complex)):
        return (1 + 0j) / np.conj(x) if np.abs(x) > 1.0 else x
    else:
        need = np.abs(x) > 1.0
        x[need] = (1 + 0j) / np.conj(x[need])
        return x

def formantFreq(n, L = 0.168, c = 340.29):
    return (2 * n - 1) * c / 4 / L

def countFormant(freq, L = 0.168, c = 340.29):
    return int(round((freq * 4 * L / c + 1) / 2))

def klattFilter(f, amp, F, bw, sr): # page 973
    C = -np.exp(2.0 * np.pi * bw / sr)
    B = -2 * np.exp(np.pi * bw / sr)
    A = 1 - B - C
    z = np.exp(2.0j * np.pi * (0.5 + (f - F) / sr))
    ampFac = (1.0 + B - C) / A * amp
    return np.abs(A / (1.0 - B / z - C / z / z)) * ampFac

def preEmphasisResponse(x, freq, sr):
    x = np.asarray(x)
    a = np.exp(-2.0 * np.pi * freq / sr)
    z = np.exp(2j * np.pi * x / sr)
    return np.abs(1 - a / z)

def preEmphasis(x, freq, sr):
    o = np.zeros(len(x))
    fac = np.exp(-2.0 * np.pi * freq / sr)
    o[0] = x[0]
    o[1:] = x[1:] - x[:-1] * fac
    return o

def deEmphasis(x, freq, sr):
    o = x.copy()
    fac = np.exp(-2.0 * np.pi * freq / sr)

    for i in range(1, len(x)):
        o[i] += o[i - 1] * fac
    return o

def toLinear(x):
    return np.power(10.0, x / 20.0)

def toLog(x):
    return np.log10(x) * 20.0

def freqToMel(x, a = 2595.0, b = 700.0):
    return a * np.log10(1.0 + x / b)

def melToFreq(x, a = 2595.0, b = 700.0):
    return (np.power(10, x / a) - 1.0) * b
