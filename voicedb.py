import numpy as np
import scipy.signal as sp
import scipy.interpolate as ipl
import pylab as pl
import os
import pickle
import sys
import traceback
from common import *

class Phone:
    def __init__(self):
        self.avgFreq = 0.0

        self.onsetPos = 0
        self.vowelPos = 0
        self.consonantPos = 0
        self.risePos = 0
        self.fallPos = 0

        self.hOFList = np.zeros((0, 0), dtype=np.float64)
        self.hPhaseList = np.zeros((0, 0), dtype=np.float64)
        self.formantList = np.zeros((0, 0), dtype=np.float64)
        self.sinusoidEnvList = np.zeros((0, 0), dtype=np.float64)
        self.noiseEnvList = np.zeros((0, 0), dtype=np.float64)
        self.sinusoidEnergyList = np.zeros(0, dtype=np.float64)
        self.noiseEnergyList = np.zeros(0, dtype=np.float64)
        self.voiced = np.zeros(0, dtype=np.bool)
        self.hfFreq = 6000.0
        self.hfBW = 300.0

class PhoneList:
    def __init__(self, type, name):
        self.name = name
        self.type = type
        self.phones = {}

    def addPhone(self, prev, curr, next, phone):
        if(not (prev, curr, next) in self.phones):
            self.phones[(prev, curr, next)] = []
        self.phones[(prev, curr, next)].append(phone)

    def maximumMatchedContext(self, prev, curr, next):
        if(curr != self.name):
            raise ValueError('Bad curr.')
        bestStatus = 0
        best = None
        for key, val in self.phones.items():
            hitL = key[0] == prev
            hitR = key[2] == next
            if(not((key[0] is None or hitL) and (key[2] is None or hitR))):
                continue
            if(hitL or hitR):
                if(hitL and hitR):
                    bestStatus = 3
                    best = key
                elif(bestStatus <= 2):
                    bestStatus = 2
                    best = key
            elif(bestStatus <= 1):
                bestStatus = 1
                best = key
        if(best is None):
            raise ValueError('Nothing matched.')
        return best, ('miss', 'valid', 'part', 'full')[bestStatus]

class VoiceDB:
    def __init__(self, path):
        self.__dict__['general'] = (('samprate', 44100), ('hopSize', 256), ('fftSize', 4096), ('nHar', 192), ('maxFormant', 6))
        self.supported = ['general', 'phone']
        self.attrs = {}
        self.path = path
        for typeName in self.supported:
            self.attrs[typeName] = {}
        self.load()

    def __del__(self):
        self.sync()

    def __getattr__(self, name):
        if(name in next(zip(*self.__dict__['general']))):
            return self.getAttr('general', name)
        else:
            return self.__dict__[name]

    def __setattr__(self, name, val):
        if(name in next(zip(*self.__dict__['general']))):
            self.setAttr(self, 'general', name, val)
        else:
            super().__setattr__(name, val)

    def attrStatus(self, typeName, name):
        if(typeName in self.attrs):
            objDict = self.attrs[typeName]
            if(not name in objDict):
                return None
            return objDict[name]["modified"]
        else:
            raise KeyError('Bad typename %s.' % (typeName))

    def addAttr(self, typeName, name, data, modify = "original", path = None):
        if(not modify in ('modified', "original", "synced")):
            if(modify == "removed"):
                raise ValueError("If you want to remove an attr, use delAttr(typeName, name).")
            raise ValueError("Invalid modify value %s.(must be 'modified' or 'original' or 'synced'.)" % (str(modify)))
        if(typeName in self.attrs):
            objDict = self.attrs[typeName]
            if(name in objDict):
                raise KeyError('Already exists %s/%s.' % (typeName, name))
            newData = {}
            newData["data"] = data
            newData["modified"] = modify
            if(path is None):
                path = os.path.join(self.path, typeName, "%s.pickle" % (name))
            newData["path"] = path
            objDict[name] = newData

    def markAttr(self, typeName, name, modify = None, path = None):
        if((not modify is None) and (not modify in ('modified', "original", "synced"))):
            if(modify == "removed"):
                raise ValueError("If you want to remove an attr, use delAttr(typeName, name).")
            raise ValueError("Invalid modify value %s.(must be 'modified' or 'original' or 'synced'.)" % (str(modify)))
        if(typeName in self.attrs):
            objDict = self.attrs[typeName]
            if(not name in objDict):
                raise KeyError('Not exists %s/%s.' % (typeName, name))
            if(not modify is None):
                objDict[name]["modified"] = modify
            if(not path is None):
                objDict[name]["path"] = path
        else:
            raise KeyError('Bad typename %s.' % (typeName))

    def setAttr(self, typeName, name, data):
        if(typeName in self.attrs):
            objDict = self.attrs[typeName]
            if(not name in objDict):
                raise KeyError('Not found %s/%s.' % (typeName, name))
            objDict[name]["modified"] = 'modified'
            objDict[name]['data'] = data
        else:
            raise KeyError('Bad typename %s.' % (typeName))

    def getAttr(self, typeName, name):
        return self.attrs[typeName][name]['data']

    def hasAttr(self, typeName, name):
        try:
            _ = self.attrs[typeName][name]
            return True
        except:
            return False

    def delAttr(self, typeName, name):
        if(typeName in self.attrs):
            objDict = self.attrs[typeName]
            if(not name in objDict):
                raise KeyError('Not exists %s/%s.' % (typeName, name))
            status = objDict[name]["modified"]
            if(status == 'synced'):
                del objDict[name]["data"]
                objDict[name]["modified"] = "removed"
            elif(status == 'original'):
                del objDict[name]
            else:
                raise ValueError('Bad status %s(%s/%s) on removing.' % (status, typeName, name))
        else:
            raise KeyError('Bad typename %s.' % (typeName))

    def load(self):
        if(os.path.exists(self.path)):
            if(not os.path.isdir(self.path)):
                raise FileNotFoundError('Invalid path "%s". (must be a directory)' % (self.path))
        else:
            os.makedirs(self.path, exist_ok = True)
        for dirpath, dirnames, filenames in os.walk(self.path):
            filteredFilenames = []
            for filename in filenames:
                if(os.path.splitext(filename)[-1] == '.pickle'):
                    filteredFilenames.append(filename)
            filenames = filteredFilenames
            del filteredFilenames
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    typeName, name, data = pickle.load(open(filepath, 'rb'))
                except:
                    print("Cannot load %s." % (filepath), file = sys.stderr)
                    raise
                if(typeName in self.supported):
                    self.addAttr(typeName, name, data, modify = 'synced', path = filepath)
                else:
                    print("Unsupported type %s in %s." % (typeName, filepath))
        for attrName, default in self.general:
            if(self.attrStatus('general', attrName) is None):
                self.addAttr('general', attrName, default)

    def sync(self):
        remove = []
        for typeName, objDict in self.attrs.items():
            for name, data in objDict.items():
                modifyStatus = data["modified"]
                if(modifyStatus == 'modified' or modifyStatus == 'original'):
                    os.makedirs(os.path.dirname(data['path']), exist_ok = True)
                    pickle.dump((typeName, name, data["data"]), open(data['path'], 'wb'))
                    data["modified"] = 'synced'
                elif(modifyStatus == 'removed'):
                    try:
                        os.remove(data['path'])
                        remove.append((typeName, name))
                    except:
                        print("Cannot do removing process on %s(%s/%s).", data['path'], typeName, name)
                        traceback.print_exc()
                elif(modifyStatus == 'synced'):
                    pass
                else:
                    raise ValueError('Bad status %s in %s/%s' % (modifyStatus, typeName, name))

        for typeName, name in remove:
            del self.attrs[typeName][name]
