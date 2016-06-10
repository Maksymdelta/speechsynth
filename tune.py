from math import *

PNTable = (
    ("C"), ("C#", "Db", "C#"),
    ("D"), ("D#", "Eb", "Eb"),
    ("E"), ("F"),
    ("F#", "Gb", "F#"), ("G"),
    ("G#", "Ab", "Ab"), ("A"),
    ("A#", "Bb", "Bb"), ("B")
)

SemitoneTable = (0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0)

def pitchToSPN(pitch_r, type = 'mixing'):
    pitch = int(pitch_r)
    s = pitch / 12
    i = pitch % 12
    if(i < 0):
        i += 12
        s -= 1
    if(type == 'mixing'):
        return PNTable[i][2 if SemitoneTable[i] else 0] + str(round(s - 2))
    elif(type == 'rising'):
        return PNTable[i][0] + str(s - 1)
    elif(type == 'falling'):
        return PNTable[i][1 if SemitoneTable[i] else 0] + str(s - 2)
    else:
        raise ValueError('Bad SPN type.')

def spnToPitch(SPN):
    if(SPN[1] == "#" or SPN[1] == "b"): # If it's semitone... XD
        pn = SPN[0:2];
        for i in range(1, 11):
            if(not SemitoneTable[i]):
                continue;
            for j in range(3):
                if(pn == PNTable[i][j]):
                    return 12 * (int(SPN[2:]) + 1) + i;
    else: # It's tritone!
        pn = SPN[0];
        for i in range(12):
            if(pn == PNTable[i][0]):
                return 12 * (int(SPN[1:]) + 1) + i;

def isSemitone(pitch_r):
    pitch = int(pitch_r);
    i = pitch % 12;
    if(i < 0):
        i += 12;
    return SemitoneTable[i];

def pitchToFreq(pitch):
    return 440.0 * pow(2.0, (pitch - 69.0) / 12.0);

def freqToPitch(freq):
    return 69.0 + 12.0 * log(2, freq / 440.0);

def addCentToFreq(freq, cent):
    return freq * pow(2.0, cent / 1200.0);

def calcCentFromFreq(f1, f2):
    return 1200.0 * log(2, f1 / f2);
