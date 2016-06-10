import numpy as np

def getFrameRange(inputLen, center, size):
    leftSize = int(size / 2)
    rightSize = size - leftSize # for odd size
    
    inputBegin = max(center - leftSize, 0)
    inputEnd = min(center + rightSize, inputLen)
    
    outBegin = -min(center - leftSize, 0)
    outEnd = outBegin + (inputEnd - inputBegin)
    
    return outBegin, outEnd, inputBegin, inputEnd

def getFrame(input, center, size):
    out = np.zeros((size), input.dtype)
    
    outBegin, outEnd, inputBegin, inputEnd = getFrameRange(len(input), center, size)
    
    out[outBegin:outEnd] = input[inputBegin:inputEnd]
    return out

def getNFrame(inputSize, hopSize):
    return inputSize / hopSize + 1 if(inputSize % hopSize != 0) else inputSize / hopSize

def roundUpToPowerOf2(v):
    return int(2 ** np.ceil(np.log2(v)))