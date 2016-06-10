import numpy as np
import scipy.signal as sp
import scipy.fftpack as fp
import pylab as pl
import pyin
import hnm
import lpc
import lpcformant
import stft
import envelope
import fecsola
import pickle
import formantMarker
import correcter
from common import *
import profile
import dbsuite
import synther
import tune
import note
import voicedb
import mfcc
import os

import ctypes
mkl_rt = ctypes.CDLL('mkl_rt.dll')
print("Default MKL thread:", mkl_rt.mkl_get_max_threads())
mkl_rt.mkl_set_dynamic(ctypes.byref(ctypes.c_int(0)))
mkl_rt.mkl_set_num_threads(ctypes.byref(ctypes.c_int(8)))
print("Current MKL thread:", mkl_rt.mkl_get_max_threads())

#dbsuite.addPhone('D:/Desktop/testdb', 'aiauea.wav', 'v', None, 'e', None, method = 'qfft')


db = voicedb.VoiceDB('D:/Desktop/testdb')

proj = note.Project()
trk = note.Track('Track 1')
proj.trackList.append(trk)
idx = trk.addNote(proj.reso, 69-10, proj.reso * 4)
_, note = trk.noteList[idx]
note.phone = 'a'
idx = trk.addNote(proj.reso*5, 69-10, proj.reso * 4)
_, note = trk.noteList[idx]
note.phone = 'i'
idx = trk.addNote(proj.reso*9, 69-10, proj.reso * 4)
_, note = trk.noteList[idx]
note.phone = 'a'
idx = trk.addNote(proj.reso*13, 69-10, proj.reso * 4)
_, note = trk.noteList[idx]

syn = synther.Processor(db)
syn(proj, trk)
