import numpy as np
from os import listdir
from os.path import isfile, join
from scipy.interpolate import interp1d
import joblib

#======================================================================
def saveVarInfo(filepath,variableNmae,Info):
    if not isfile(filepath):
        specifications={}
        specifications[variableNmae]=Info
        joblib.dump(specifications,filepath)
    else:
        specifications=joblib.load(filepath)
        specifications[variableNmae]=Info
        joblib.dump(specifications,filepath)
def getVarInfo(filepath,variableNmae):
    if not isfile(filepath): raise Exception("File not found")
    specifications=joblib.load(filepath)
    if specifications.get(variableNmae) is None: raise Exception("Variable Info not Saved")
    return specifications[variableNmae]
#======================================================================
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

def getWavelengthSet(spectralWavelength,startW=None,endW=None):
    if startW is None:
        s=[swc[0] for swc in spectralWavelength]
        startW=np.max(s)
    if endW is None:
        e=[swc[-1] for swc in spectralWavelength]
        endW=np.min(e)
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

def readSpectralLib(spectralLibPath,startW=None,endW=None):
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

    spectralWavelengthSet= getWavelengthSet(spectralWavelength,startW=startW,endW=endW)
    return spectralWavelength,spectralIF,mineralNames,spectralWavelengthSet #spectralFiles,
#======================================================================
def nextCharIndex(content,searchWord):
    searchWordSize=len(searchWord)
    for i in range(searchWordSize,len(content)-searchWordSize):
        if content[i-searchWordSize:i]==searchWord:
            break
    return i
def getStringUntil(content,startIndex,endChar):
    outString,i='',startIndex
    while True:
        newChar=content[i]
        if newChar==endChar:
            break
        else:
            outString+=newChar
        i=i+1
    return outString

def readImage(hdrFile,imageFile):
    hdrFileContent=open(hdrFile,'r').read()

    i=nextCharIndex(hdrFileContent,'lines   =')
    LINES=getNum(getStringUntil(hdrFileContent,i,'\n'))
    i=nextCharIndex(hdrFileContent,'samples =')
    LINE_SAMPLES=getNum(getStringUntil(hdrFileContent,i,'\n'))
    i=nextCharIndex(hdrFileContent,'bands   =')
    BANDS=getNum(getStringUntil(hdrFileContent,i,'\n'))
    print(LINES,LINE_SAMPLES,BANDS)

    i=nextCharIndex(hdrFileContent,'interleave = ')
    interleave=getStringUntil(hdrFileContent,i,'\n')
    # interleave='bsq'

    i=nextCharIndex(hdrFileContent,'wavelength units = ')
    wavelength_units=getStringUntil(hdrFileContent,i,'\n')

    i=nextCharIndex(hdrFileContent,'wavelength = {')
    wavelengthString=getStringUntil(hdrFileContent,i,'}')

    print(interleave,wavelength_units)
    # print(wavelengthString)

    wavelengths=[]
    for w in wavelengthString.split(','):
        if wavelength_units=='Micrometers':
            wavelengths.append(round(getNum(w),5))
        elif wavelength_units=='Nanometers':
            wavelengths.append(round(getNum(w)/1000,5))

    if interleave=='bsq':
        image = np.fromfile(open(imageFile, 'rb'), np.dtype('float32')).reshape((BANDS,LINES,LINE_SAMPLES))
        print(image.shape)
        print("image read as bsq")
    if interleave=='bil':
        image=[]
        tempMat=np.fromfile(open(imageFile, 'rb'), np.dtype('float32')).reshape(BANDS*LINES,-1)    
        for l in range(BANDS):
            thisBand=[]
            for b in range(l,len(tempMat),BANDS):
                thisBand.append(tempMat[b])
            image.append(thisBand)
        image=np.array(image)
        print("image read as bil")

    imageFrame=np.full((LINES,LINE_SAMPLES),False)
    for x1 in range(LINES):
        for y1 in range(LINE_SAMPLES):
            if image[0,x1,y1]==65535:            
                imageFrame[x1,y1]=True

    return wavelengths,image,LINES,LINE_SAMPLES,BANDS,imageFrame