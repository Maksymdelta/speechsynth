import bisect
import match
from common import *

class Note:
    def __init__(self, k, l):
        self.key = k
        self.length = l
        self.phone = 'a'
        self.lang = ''

    @property
    def decoedPhone(self):
        if(self.lang):
            return "mltl_%s@%s"% (self.phone, self.lang)
        return self.phone

class Track:
    def __init__(self, name):
        self.noteList = []
        self.name = name

    def getNote(self, p, k = None):
        i = bisect.bisect_right(next(zip(*self.noteList)), p)
        ret = []
        for iNote, (pos, note) in enumerate(self.noteList[:i]):
            if(pos <= p and pos + note.length > p and note.key == k):
                ret.append(iNote)
        return ret

    def addNote(self, p, k, l):
        if(len(self.noteList) == 0):
            self.noteList.append((p, Note(k, l)))
            return 0
        i = bisect.bisect_right(next(zip(*self.noteList)), p)
        self.noteList.insert(i, (p, Note(k, l)))
        return i

    def removeNote(self, iNote):
        del self.noteList[iNote]

    def hasOverlapped(self):
        beginList = next(zip(*self.noteList))
        endList = [pos + note.length for pos, note in self.noteList]
        overlapSet = set()
        for iNote, (posA, noteA) in enumerate(self.noteList):
            overlapped = False
            endA = posA + noteA.length
            for jNote, (posB, noteB) in enumerate(self.noteList):
                if(posB >= endA):
                    break
                if(iNote != jNote and posB + noteB.length > posA):
                    overlapSet.add(jNote)
                    overlapped = True
            if(overlapped):
                overlapSet.add(iNote)
        return list(overlapSet)

    @property
    def beginPos(self):
        return self.noteList[0][0] if(self.noteList) else 0

    @property
    def endPos(self):
        if(not self.noteList):
            return 0
        return max([pos + note.length for pos, note in self.noteList])

class Project:
    def __init__(self):
        self.reso = 960
        self.tempoList = match.Match(allowSameNeighbor = False, first = (0, 120.0))
        self.beatList = match.Match(allowSameNeighbor = False, first = (0, (4, 4)))
        self.trackList = []
        self.tempoCache = None
        self.beatCache = None

    def addTempo(self, pos, val):
        assert(val > 0)
        pos = int(pos)
        val = float(val)
        self.tempoList.add(pos, val)
        self.tempoCache = None

    def removeTempo(self, iTempo):
        self.tempoList.remove(iTempo)
        self.tempoCache = None

    def tempoAtTick(self, tick):
        return self.tempoList.query(tick)

    def tempoAtTime(self, time):
        if(len(self.tempoList) == 1 or time <= 0.0):
            return 0
        self.updateTempoCache()
        return match.Match.queryCore(self.tempoCache, time)

    def addBeat(self, sect, val):
        assert(len(val) == 2)
        pos = int(pos)
        val = (int(val[0]), int(val[1]))
        if(val[0] <= 0 or val[1] <= 0):
            raise ValueError('Bad beat.')
        self.beatList.add(sect, val)
        self.beatCache = None

    def removeBeat(self, iBeat):
        self.beatList.remove(iBeat)
        self.beatCache = None

    def beatAtSect(self, sect):
        return self.beatList.query(sect)

    def updateTempoCache(self):
        if(self.tempoCache is None):
            if(not self.tempoList):
                self.tempoCache = []
                return
            cache = []
            time = 0.0
            lastPos, lastVal = self.tempoList[0]
            for pos, val in self.tempoList[1:]:
                cache.append(time)
                time += (pos - lastPos) / self.reso / lastVal * 60.0
                lastPos, lastVal = pos, val
            cache.append(time)
            self.tempoCache = cache

    def updateBeatCache(self):
        if(self.beatCache is None):
            if(not self.beatList):
                self.beatCache = []
                return
            cache = []
            tick = 0
            lastPos, (lastN, lastD) = self.beatList[0]
            for pos, (n, d) in self.tempoList[1:]:
                cache.append(tick)
                tick += int((pos - lastPos) * lastN * self.reso * 4 / lastD)
                lastPos, lastN, lastD = pos, n, d
            cache.append(tick)
            self.beatCache = cache

    def tickToTime(self, tick):
        if(len(self.tempoList) == 1 or tick <= 0):
            return tick / self.reso / self.tempoList[0][1] * 60.0
        self.updateTempoCache()
        iTempo = self.tempoAtTick(tick)
        pos, val = self.tempoList[iTempo]
        return self.tempoCache[iTempo] + (tick - pos) / self.reso / val * 60.0

    def timeToTick(self, time):
        if(len(self.tempoList) == 1 or time <= 0):
            return time / 60.0 * self.reso * self.tempoList[0][1]
        self.updateTempoCache()
        iTempo = self.tempoAtTime(time)
        pos, val = self.tempoList[iTempo]
        return pos + (time - self.tempoCache[iTempo]) / 60.0 * self.reso * val

    def mbtToTick(self, m, b, t):
        if(len(self.beatList) == 1 or m <= 0):
            pos, (n, d) = self.beatList[0]
            return m * self.reso * 4 * n / d + b * self.reso * 4 / d + t

        self.updateBeatCache()
        iBeat = self.beatList.query(m)
        pos, (n, d) = self.beatList[iBeat]
        return self.beatCache[iBeat] + (m - pos) * self.reso * 4 * n / d + b * self.reso * 4 / d + t

    def tickToMBT(self, tick):
        sym = -1 if(tick < 0) else 0
        if(len(self.tempoList) == 1 or tick <= 0):
            pos, (n, d) = self.beatList[0]
            sect = int(tick / (self.reso * 4 * n / d))
            tick = tick % (self.reso * 4 * n / d)
            if(sym == -1 and tick):
                sect += sym
            beat = int(tick / (self.reso * 4 / d))
            tick = tick % (self.reso * 4 / d)
            return sect, beat, tick
        self.updateBeatCache()
        iBeat = match.Match.queryCore(self.beatCache, tick)
        pos, (n, d) = self.beatList[iBeat]
        tick -= self.beatCache[iBeat]
        sect = int(tick / (self.reso * 4 * n / d))
        tick = tick % (self.reso * 4 * n / d)
        if(sym == -1 and tick):
            sect += sym
        beat = int(tick / (self.reso * 4 / d))
        tick = tick % (self.reso * 4 / d)
        sect += sym
        return sect + pos, beat, tick
