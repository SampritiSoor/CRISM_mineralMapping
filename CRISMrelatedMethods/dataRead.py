import numpy as np
from os import listdir
from os.path import isfile, join
from scipy.interpolate import interp1d
from math import factorial
from sklearn.preprocessing import StandardScaler

def getNum(test_string):
    out_string=''
    for i in range(len(test_string)):
        if test_string[i]=='-' or test_string[i]=='.' or test_string[i].isdigit():
            out_string=out_string+test_string[i]
    try:
        if '.' in out_string:
            return np.float32(out_string)
        else:
            return np.int32(out_string)
    except:
        return np.Inf

def getWavelengthSet(spectralWavelength,startW=1,endW=2.6):
    startW,endW=np.float32(startW),np.float32(endW)
    wavelengthSet=[]
    for swc in spectralWavelength:
        wavelengthSet.extend(swc)
    wavelengthSet=list(set(wavelengthSet))
    wavelengthSet.sort()
    wavelengthSet=np.float32(wavelengthSet)
    wl1Start=len([1 for w in wavelengthSet if w<startW])
    wl1End=len([1 for w in wavelengthSet if w<=endW])
    return wavelengthSet[wl1Start:wl1End]

def readSpectralLib(spectralLibPath,startW=1,endW=2.6):
    spectralFiles = [f for f in listdir(spectralLibPath) if isfile(join(spectralLibPath, f))]

    Y=3 # 1: ratioed I/F ; 3:numarator I/F ; 5:denominator I/F
    spectralWavelength,spectralIF=[],[]
    mineralNames={}
    for file_index in range(len(spectralFiles)):
        mineralNames[file_index]=spectralFiles[file_index][15:-4]
        thisMinarel_wavelength,thisMinarel_IF=[],[]
        fileContent=open(spectralLibPath+spectralFiles[file_index],'r')
        while True:
            thisLine=fileContent.readline()
            if not thisLine:
                break
            if getNum(thisLine.split(',')[Y])<10: # exclude wavelengths with not acceptable IFs
                thiswv=getNum(thisLine.split(',')[0])
                thisMinarel_wavelength.append(getNum(thisLine.split(',')[0]))
                thisMinarel_IF.append(getNum(thisLine.split(',')[Y]))
        spectralWavelength.append(thisMinarel_wavelength)
        spectralIF.append(thisMinarel_IF)

    spectralWavelengthSet= getWavelengthSet(spectralWavelength,startW,endW)
    return spectralWavelength,spectralIF,spectralFiles,mineralNames,spectralWavelengthSet

import scipy.io as io
import random
def getPlebaniData(path,plebaniMICAclass,spectralWavelengthSet,perLabelData=100,MICA=True):
    trdr_wavelengths = [
    1.021, 1.02755, 1.0341, 1.04065, 1.0472, 1.05375, 1.0603, 1.06685,1.07341, 1.07996, 1.08651, 1.09307, 1.09962, 1.10617, 1.11273, 1.11928,1.12584, 1.13239, 1.13895, 1.14551, 1.15206, 1.15862, 
    1.16518, 1.17173, 1.17829, 1.18485, 1.19141, 1.19797, 1.20453, 1.21109, 1.21765, 1.22421, 1.23077, 1.23733, 1.24389, 1.25045, 1.25701, 1.26357, 1.27014, 1.2767, 1.28326, 1.28983, 1.29639, 
    1.30295, 1.30952, 1.31608, 1.32265, 1.32921, 1.33578, 1.34234, 1.34891, 1.35548, 1.36205, 1.36861, 1.37518, 1.38175, 1.38832, 1.39489, 1.40145, 1.40802, 1.41459, 1.42116, 1.42773, 1.43431, 
    1.44088, 1.44745, 1.45402, 1.46059, 1.46716, 1.47374, 1.48031, 1.48688, 1.49346, 1.50003, 1.50661, 1.51318, 1.51976, 1.52633, 1.53291, 1.53948, 1.54606, 1.55264, 1.55921, 1.56579, 1.57237, 
    1.57895, 1.58552, 1.5921, 1.59868, 1.60526, 1.61184, 1.61842, 1.625, 1.63158, 1.63816, 1.64474, 1.65133, 1.65791, 1.66449, 1.67107, 1.67766, 1.68424, 1.69082, 1.69741, 1.70399, 1.71058, 
    1.71716, 1.72375, 1.73033, 1.73692, 1.74351, 1.75009, 1.75668, 1.76327, 1.76985, 1.77644, 1.78303, 1.78962, 1.79621, 1.8028, 1.80939, 1.81598, 1.82257, 1.82916, 1.83575, 1.84234, 1.84893, 
    1.85552, 1.86212, 1.86871, 1.8753, 1.8819, 1.88849, 1.89508, 1.90168, 1.90827, 1.91487, 1.92146, 1.92806, 1.93465, 1.94125, 1.94785, 1.95444, 1.96104, 1.96764, 1.97424, 1.98084, 1.98743, 
    1.99403, 2.00063, 2.00723, 2.01383, 2.02043, 2.02703, 2.03363, 2.04024, 2.04684, 2.05344, 2.06004, 2.06664, 2.07325, 2.07985, 2.08645, 2.09306, 2.09966, 2.10627, 2.11287, 2.11948, 2.12608, 
    2.13269, 2.1393, 2.1459, 2.15251, 2.15912, 2.16572, 2.17233, 2.17894, 2.18555, 2.19216, 2.19877, 2.20538, 2.21199, 2.2186, 2.22521, 2.23182, 2.23843, 2.24504, 2.25165, 2.25827, 2.26488, 2.27149, 
    2.2781, 2.28472, 2.29133, 2.29795, 2.30456, 2.31118, 2.31779, 2.32441, 2.33102, 2.33764, 2.34426, 2.35087, 2.35749, 2.36411, 2.37072, 2.37734, 2.38396, 2.39058, 2.3972, 2.40382, 2.41044, 2.41706, 
    2.42368, 2.4303, 2.43692, 2.44354, 2.45017, 2.45679, 2.46341, 2.47003, 2.47666, 2.48328, 2.4899, 2.49653, 2.50312, 2.50972, 2.51632, 2.52292, 2.52951, 2.53611, 2.54271, 2.54931, 2.55591, 2.56251, 
    2.56911, 2.57571, 2.58231, 2.58891, 2.59551, 2.60212, 2.60872, 2.61532, 2.62192, 2.62853, 2.63513, 2.64174, 2.64834, 2.80697, 2.81358, 2.8202, 2.82681, 2.83343, 2.84004, 2.84666, 2.85328, 2.85989,
    2.86651, 2.87313, 2.87975, 2.88636, 2.89298, 2.8996, 2.90622, 2.91284, 2.91946, 2.92608, 2.9327, 2.93932, 2.94595, 2.95257, 2.95919, 2.96581, 2.97244, 2.97906, 2.98568, 2.99231, 2.99893, 3.00556, 
    3.01218, 3.01881, 3.02544, 3.03206, 3.03869, 3.04532, 3.05195, 3.05857, 3.0652, 3.07183, 3.07846, 3.08509, 3.09172, 3.09835, 3.10498, 3.11161, 3.11825, 3.12488, 3.13151, 3.13814, 3.14478, 3.15141,
    3.15804, 3.16468, 3.17131, 3.17795, 3.18458, 3.19122, 3.19785, 3.20449, 3.21113, 3.21776, 3.2244, 3.23104, 3.23768, 3.24432, 3.25096, 3.2576, 3.26424, 3.27088, 3.27752, 3.28416, 3.2908, 3.29744, 
    3.30408, 3.31073, 3.31737, 3.32401, 3.33066, 3.3373, 3.34395, 3.35059, 3.35724, 3.36388, 3.37053, 3.37717, 3.38382, 3.39047, 3.39712, 3.40376, 3.41041, 3.41706, 3.42371, 3.43036, 3.43701, 3.44366,
    3.45031, 3.45696, 3.46361, 3.47026, 3.47692]
    trdr_WavelengthSet=np.array(trdr_wavelengths) #[:len([w for w in trdr_wavelengths if w<=Wend])]

    data_ratioed=io.loadmat(path)
    data=data_ratioed
    label_dict={}
    for i in range(data['pixlabs'].shape[0]):
        if label_dict.get(data['pixlabs'][i][0]) is None:
            label_dict[data['pixlabs'][i][0]]=[data['pixspec'][i]]
        else:
            label_dict[data['pixlabs'][i][0]].append(data['pixspec'][i])
    
    data_X,data_Y=[],[]
    for l in label_dict:
        if MICA and plebaniMICAclass.get(l) is None: continue
        
        if len(label_dict[l])>=perLabelData:
            for i in random.sample(list(np.arange(len(label_dict[l]))),perLabelData):
                if np.isclose(np.std(label_dict[l][i][:len(trdr_WavelengthSet)]),0): continue
                if MICA: 
                    intp=interp1d(trdr_WavelengthSet,label_dict[l][i][:len(trdr_WavelengthSet)], kind='linear')
                    data_X.append(intp(spectralWavelengthSet))
                    data_Y.append(plebaniMICAclass[l])
                else: 
                    data_X.append(label_dict[l][i][:len(trdr_WavelengthSet)])
                    data_Y.append(l)
        else:
            count=0
            doBreak=False
            while True:
                for i in range(len(label_dict[l])):
                    if np.isclose(np.std(label_dict[l][i][:len(trdr_WavelengthSet)]),0): continue
                    if MICA: 
                        intp=interp1d(trdr_WavelengthSet,label_dict[l][i][:len(trdr_WavelengthSet)], kind='linear')
                        data_X.append(intp(spectralWavelengthSet))
                        data_Y.append(plebaniMICAclass[l])
                    else: 
                        data_X.append(label_dict[l][i][:len(trdr_WavelengthSet)])
                        data_Y.append(l)
                    count+=1
                    if count==perLabelData:
                        doBreak=True
                        break
                if doBreak: break
    X=np.array(data_X)
    Y=np.array(data_Y)
    return X,Y