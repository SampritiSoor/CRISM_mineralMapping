import numpy as np
import random
from scipy.interpolate import interp1d
from math import factorial,atan2, pi, degrees, isclose
from sklearn.preprocessing import StandardScaler

#===========================================================
def getShiftNoised(sourceSpectra,maxprcn=60,initialDev=6):
    noiseX=[sourceSpectra[0]*((1-((initialDev/2)/100))+(random.random()*(initialDev/100)))]
    for l in range(1,len(sourceSpectra)):
        chngOrg=((sourceSpectra[l]-sourceSpectra[l-1])/sourceSpectra[l-1]) 
        frac=chngOrg*((1-(maxprcn/100))+ (2*maxprcn*random.random()/100))
        noiseX.append(noiseX[-1]*(1+frac))
    return np.array(noiseX)

def getContiNoised(aSpectra,targetWL=None,nMeans=1,noiseStd=0,returnNoise=False):
    if targetWL is None:
        targetWL=np.arange(aSpectra.shape[0])
    noise=np.random.normal(np.mean(aSpectra),noiseStd,aSpectra.shape[0])
    gNoise=np.ones(aSpectra.shape)
    for i in range(nMeans):
        mean = targetWL[np.random.randint(aSpectra.shape[0])]
        std = .65 #/nMeans
        gCurve = (1/(std * np.sqrt(2 * np.pi)) * np.exp( - (targetWL - mean)**2 / (2 * std**2)))
        gNoise+=gCurve  +noise
    if returnNoise:
        return gNoise*aSpectra,gNoise
    else: 
        return gNoise*aSpectra
#===========================================================
def savitzky_golay(y, window_size=11, order=2, deriv=0, rate=1):
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
#     except ValueError, msg: SAM
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

def removeSpikes(spectra,stdRatio=0,windowSize=5,weighted=False):
    cv=np.std(spectra)/np.mean(spectra)
    windowCv=[]
    windowMean=[]
    for i in range(spectra.shape[0]):
        if i<int(windowSize/2):
            thisCv=np.std(spectra[:i+int(windowSize/2)+1])/np.mean(spectra[:i+int(windowSize/2)+1])
            thisMean=np.mean(spectra[:i+int(windowSize/2)+1])
        elif spectra.shape[0]-1-i<int(windowSize/2):
            thisCv=np.std(spectra[i-int(windowSize/2):])/np.mean(spectra[i-int(windowSize/2):])
            thisMean=np.mean(spectra[i-int(windowSize/2):])
        else:
            thisCv=np.std(spectra[i-int(windowSize/2):i+int(windowSize/2)+1])/np.mean(spectra[i-int(windowSize/2):i+int(windowSize/2)+1])
            thisMean=np.mean(spectra[i-int(windowSize/2):i+int(windowSize/2)+1])
        windowCv.append(thisCv)
        windowMean.append(thisMean)
    windowCv=np.array(windowCv)   
    windowMean=np.array(windowMean)
    cutoff= np.mean(windowCv)+stdRatio*(np.max(windowCv)-np.mean(windowCv))  
    smSpectra=np.where(windowCv>cutoff,windowMean,spectra)
    return smSpectra

def getSmoothedData(data,SGwindowSize=11, order=2, RSwindowSize=5, RSstdRatio=0):
    SMdata=[]
    for i in range(data.shape[0]):
        SMdata.append(removeSpikes(savitzky_golay(data[i,:],window_size=SGwindowSize),windowSize=RSwindowSize,stdRatio=RSstdRatio) )
    return np.array(SMdata)
#===========================================================
def angle(C, B=(0,0), A=(1,0), nintyDegree=False):
    Ax, Ay = A[0]-B[0], A[1]-B[1]
    Cx, Cy = C[0]-B[0], C[1]-B[1]
    a = atan2(Ay, Ax)
    c = atan2(Cy, Cx)
    if a < 0: a += pi*2
    if c < 0: c += pi*2
    deg= degrees((pi*2 + c - a) if a > c else (c - a))
    if not nintyDegree:
        return deg
    else:
        return deg if deg<=90 else deg-360
def getUpperHullPoints_CH(spectra,targetWL=None,doPrint=False):
    if targetWL is None: targetWL=np.arange(spectra.shape[0])
    points=list(zip(targetWL,spectra)) 
    
    hullPoints=[]
    stack=[points[0],points[1]]
    for i in range(2,len(points)):
        while True:
            if angle(stack[-2],stack[-1],points[i])>=180:
                if doPrint: print('here1',stack[-2:],points[i],angle(stack[-2],stack[-1],points[i]))
                stack.append(points[i])
                break
            else:
                if doPrint: print('here2')
                stack.pop(-1)
                if len(stack)==1:
                    stack.append(points[i])
                    break
        if doPrint:
            print(i)
            plt.figure(figsize=(10,2))
            plt.plot([s[0] for s in stack],[s[1] for s in stack])
            plt.scatter([s[0] for s in stack],[s[1] for s in stack],color='r')
            plt.plot(targetWL[:i+1],spectra[:i+1])
            plt.show()
    return [s[0] for s in stack],[s[1] for s in stack]
def getUpperCH(sourceSpectra,targetWL=None,doPrint=False):
    if targetWL is None: targetWL=np.arange(spectra.shape[0])
    UHx,UHy=getUpperHullPoints_CH(sourceSpectra,targetWL=targetWL,doPrint=doPrint)
    UHinterpFunc=interp1d(UHx,UHy, kind='linear')
    interpUH=UHinterpFunc(targetWL)
    return interpUH
def getContinuumRemovedSpectra_CH(sourceSpectra,targetWL=None):
    if targetWL is None: targetWL=np.arange(sourceSpectra.shape[0])
    continuumRemoved=sourceSpectra/getUpperCH(sourceSpectra,targetWL)
    return continuumRemoved
def getContinuumRemovedData(data,targetWL=None):
    if targetWL is None: targetWL=np.arange(data[0].shape[0])
    CRdata=[]
    for i in range(data.shape[0]):
        CRdata.append(getContinuumRemovedSpectra_CH(data[i,:],targetWL))
    return np.array(CRdata)
#===========================================================
def getStandardScaledSpectra(spectra):
    targetSpectra=spectra.reshape(-1,1)
    scaler=StandardScaler()
    targetSpectra=scaler.fit_transform(targetSpectra)
    targetSpectra=(targetSpectra.reshape(1,-1))[0]
    return targetSpectra
def getStandardScaledData(data):
    Ndata=[]
    for i in range(data.shape[0]):
        Ndata.append(getStandardScaledSpectra(data[i,:]))
    return np.array(Ndata)
#==========================================================
def getAbsorptionPositions(prepSpectra):
    maxV=np.max(prepSpectra)
    segments=[]
    segStarted=False
    for i in range(prepSpectra.shape[0]):
        if segStarted:
            if np.isclose(maxV,prepSpectra[i]):
                thisSegment['end']=i-1
                thisSegment['min']=thisSegMin[0]
                segments.append(thisSegment)
                if i<prepSpectra.shape[0]-1:
                    thisSegment={'start':i+1}
                    thisSegMin=(i+1,prepSpectra[i+1])
                if i<prepSpectra.shape[0]-1 and np.isclose(maxV,prepSpectra[i+1]):
                    segStarted=False
            else:
                if thisSegMin[1]>prepSpectra[i]:
                    thisSegMin=(i,prepSpectra[i])
        else:
            if i<prepSpectra.shape[0]-1 and np.isclose(maxV,prepSpectra[i]) and maxV>prepSpectra[i+1] and not segStarted:
                segStarted=True
                thisSegment={'start':i+1}
                thisSegMin=(i+1,prepSpectra[i+1])
    return [(S['start']-1,S['min'],S['end']+1) for S in segments]

def getbanddepth(iVal,mVal,jVal,iPos,mPos,jPos):
    return iVal - mVal + (jVal-iVal)*((mPos-iPos)/(jPos-iPos))
def getMAD(A):
    med=np.median(A)
    dev=A-med
    absdev=np.abs(dev)
    return np.median(absdev)

def getDiversePositions(sourceIF,sourceWV,targetWV,positionsCount=300,minimumDeepsize=3):
    libSpectras=[]
    for m in range(len(sourceIF)):
        pixelInterPolFunc1=interp1d(sourceWV[m],sourceIF[m], kind='linear')
        prepSpectra=getStandardScaledSpectra(getContinuumRemovedSpectra_CH(removeSpikes(savitzky_golay(pixelInterPolFunc1(targetWV)))))
        libSpectras.append(prepSpectra)
        
    D=[]
    for l in range(len(libSpectras)):
        D.extend([A for A in getAbsorptionPositions(libSpectras[l]) if (A[2]-A[0])>minimumDeepsize])
    
    diffList=[]
    for d in D:
        thisMdiffs=[]
        for k in range(len(libSpectras)):
            thisMdiffs.append(getbanddepth(libSpectras[k][d[0]],libSpectras[k][d[1]],libSpectras[k][d[2]],d[0],d[1],d[2]))
        thisMdiffs=np.array(thisMdiffs)
        thisrangeDiffScaled=(thisMdiffs-min(thisMdiffs))/(max(thisMdiffs)-min(thisMdiffs))
        diffList.append([getMAD(thisrangeDiffScaled)/np.median(thisrangeDiffScaled),d[0],d[1],d[2]])
    diffList.sort(key= lambda x:x[0],reverse=True)

    deepsFinal={}
    removeDeep=[]
    for d in range(len(diffList)):
        checkMinSize=False
        if diffList[d][3]-diffList[d][1]<minimumDeepsize:
            removeDeep.append(d)
            checkMinSize=True
        if not checkMinSize:
            checkOverlap=False
            for i in range(diffList[d][1]-3,diffList[d][1]+4):
                for m in range(diffList[d][2]-2,diffList[d][2]+3):
                    for j in range(diffList[d][3]-3,diffList[d][3]+4):
                        if deepsFinal.get((i,m,j)) is not None: 
                            checkOverlap=True
                            break
            if checkOverlap:
                removeDeep.append(d)
        if not checkMinSize and not checkOverlap:
            deepsFinal[(diffList[d][1],diffList[d][2],diffList[d][3])]='dummy'
    removeDeep.reverse()
    for r in removeDeep:
        diffList.pop(r)
    return [diffList[d][1:] for d in range(min(len(diffList),positionsCount))]
#==========================================================
def getFeature(d,rPos,feature='banddepth'):
    dd=[]
    for r in rPos:
        if feature=='banddepth':
            dd.append(  getbanddepth(d[r[0]],d[r[1]],d[r[2]],r[0],r[1],r[2])  )
    return np.array(dd)
def getFeatureData(D,rPos,feature='banddepth'):
    D_r=[]
    for d in range(D.shape[0]):
        D_r.append(getFeature(D[d],rPos,feature=feature))
    return np.array(D_r)
#==========================================================