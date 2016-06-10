import numpy as np
import numba as nb

def au_to_b(a, u):
    return a / u - a;

def normalized_pdf(a, b, begin, end, number):
    x = np.arange(0, number, dtype = np.float64) * ((end - begin) / number)
    v = np.power(x, a - 1.0) * np.power(1.0 - x, b - 1.0)
    for i in range(2, len(v) + 1):
        i = len(v) - i
        if(v[i] < v[i + 1]):
            v[i] = v[i + 1]
    return v / np.sum(v)

class Observer():
    def __init__(self, init, frm, to, transProb):
        self.init = init
        self.frm = frm
        self.to = to
        self.transProb = transProb
    
    def decodeViterbi(self, obsProb):
        nState = len(self.init)
        nFrame = len(obsProb)
        nTrans = len(self.transProb)

        scale = np.zeros((nFrame), dtype = np.float64)

        if(len(obsProb) < 1): # too short
            return np.array([], dtype = np.int)

        delta = np.zeros((nState), dtype = np.float64)
        oldDelta = np.zeros((nState), dtype = np.float64)
        psi = np.zeros((nFrame, nState), dtype = np.int) # matrix of remembered indices of the best transitions
        path = np.ndarray((nFrame), dtype = np.int) # the final output path
        path.fill(nState - 1)

        # init first frame
        oldDelta = self.init * obsProb[0][:len(self.init)]
        deltaSum = np.sum(oldDelta)

        scale[0] = 1.0 / deltaSum

        # rest of forward step

        for iFrame in range(1, nFrame):
            transRange = np.arange(nTrans)
            fromState = self.frm[transRange]
            toState = self.to[transRange]
            currTransProb = self.transProb[transRange]
            currValue = oldDelta[fromState] * currTransProb

            for iTrans in range(nTrans):
                ts = toState[iTrans]
                if(currValue[iTrans] > delta[ts]):
                    delta[ts] = currValue[iTrans] # will be multiplied by the right obs later
                    psi[iFrame][ts] = fromState[iTrans]

            delta *= obsProb[iFrame][:len(self.init)]
            deltaSum = np.sum(delta)

            if(deltaSum > 0):
                oldDelta = delta / deltaSum
                delta.fill(0)
                scale[iFrame] = 1.0 / deltaSum
            else:
                print("WARNING: Viterbi has been fed some zero probabilities at frame %d." % (iFrame), file = sys.stderr)
                oldDelta.fill(1.0 / nState)
                delta.fill(0)
                scale[iFrame] = 1.0

        # init backward step
        bestStateIdx = np.argmax(oldDelta)
        bestValue = oldDelta[bestStateIdx]
        path[-1] = bestStateIdx
        # rest of backward step
        for iFrame in reversed(range(nFrame - 1)):
            path[iFrame] = psi[iFrame + 1][path[iFrame + 1]]
        return path
