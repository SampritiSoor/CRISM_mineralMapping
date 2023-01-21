import numpy as np
import random
from scipy.interpolate import interp1d
from math import factorial,atan2, pi, degrees, isclose
from sklearn.preprocessing import StandardScaler,MinMaxScaler

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
        std = random.randint(2, 5)*.25
        gCurve = (1/(std * np.sqrt(2 * np.pi)) * np.exp( - (targetWL - mean)**2 / (2 * std**2)))
        gNoise+=gCurve  +noise
    if returnNoise:
        return gNoise*aSpectra,gNoise
    else: 
        return gNoise*aSpectra
#===========================================================
def savitzky_golay(y, window_size=11, order=2, deriv=0, rate=1):
    window_size,order=np.int32(window_size),np.int32(order)
    if  window_size < 3:  window_size=3
    if window_size % 2 == 0:  window_size+=1
    if window_size < order + 2: window_size=order + 2
        
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

def getSmoothedData(data,SGwindowSize=11, SGorder=2, RSwindowSize=5, RSstdRatio=0, prep='smRS',mask=None):
    if mask is None: mask=np.full(data.shape,False).ravel()
    SMdata=np.ones(data.shape)
    for i in range(data.shape[0]):
        if mask[i]: continue
        if prep=='smRS': SMdata[i]=removeSpikes(savitzky_golay(data[i],window_size=SGwindowSize,order=SGorder),windowSize=RSwindowSize,stdRatio=RSstdRatio)
        if prep=='sm': SMdata[i]=savitzky_golay(data[i],window_size=SGwindowSize,order=SGorder)
        if prep=='RS': SMdata[i]=removeSpikes(data[i],windowSize=RSwindowSize,stdRatio=RSstdRatio)
        if prep=='RSsm': SMdata[i]=savitzky_golay(removeSpikes(data[i],windowSize=RSwindowSize,stdRatio=RSstdRatio),window_size=SGwindowSize,order=SGorder)
    return SMdata
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
def getContinuumRemovedData(data,targetWL=None,mask=None):
    if mask is None: mask=np.full(data.shape,False).ravel()
    if targetWL is None: targetWL=np.arange(data[0].shape[0])
    CRdata=np.ones(data.shape)
    for i in range(data.shape[0]):
        if mask[i]: continue
        CRdata[i]=getContinuumRemovedSpectra_CH(data[i],targetWL)
    return CRdata
#===========================================================
def getStandardScaledSpectra(spectra):
    targetSpectra=spectra.reshape(-1,1)
    scaler=StandardScaler()
    targetSpectra=scaler.fit_transform(targetSpectra)
    targetSpectra=(targetSpectra.reshape(1,-1))[0]
    return targetSpectra
def getStandardScaledData(data,mask=None):
    if mask is None: mask=np.full(data.shape,False).ravel()
    Ndata=np.ones(data.shape)
    for i in range(data.shape[0]):
        if mask[i]: continue
        Ndata[i]=getStandardScaledSpectra(data[i])
    return np.array(Ndata)
#==========================================================
def getAbsorptionPositions(prepSpectra,minimumDeepBands=3,areaCutOff=.01): #,weight=False
    maxV=np.max(prepSpectra)
    totalArea=np.sum([maxV-p for p in prepSpectra])
    segments=[]
    segStarted=False
    for i in range(prepSpectra.shape[0]):
        if segStarted:
            if np.isclose(maxV,prepSpectra[i]):
                thisSegment['end']=i
                thisSegment['min']=thisSegMin[0]
                addNewSegment=True
#                 if (thisSegment['end']-thisSegment['start'])<=minimumDeepBands:
#                     addNewSegment=False
                if (thisSegment['area']/totalArea)<areaCutOff:
                    addNewSegment=False
                if addNewSegment:
                    segments.append(thisSegment)
                if i<prepSpectra.shape[0]-1:
                    thisSegment={'start':i,'area':(maxV-prepSpectra[i])}
                    thisSegMin=(i,prepSpectra[i])
                if i<prepSpectra.shape[0]-1 and np.isclose(maxV,prepSpectra[i+1]):
                    segStarted=False
            else:
                thisSegment['area']+=(maxV-prepSpectra[i])
                if thisSegMin[1]>prepSpectra[i]:
                    thisSegMin=(i,prepSpectra[i])
        else:
            if i<prepSpectra.shape[0]-1 and np.isclose(maxV,prepSpectra[i]) and maxV>prepSpectra[i+1] and not segStarted:
                segStarted=True
                thisSegment={'start':i,'area':(maxV-prepSpectra[i])}
                thisSegMin=(i,prepSpectra[i])
    return [(S['start'],S['min'],S['end'], S['area']) for S in segments]

def getbanddepth(iVal,mVal,jVal,iPos,mPos,jPos):
    return iVal - mVal + (jVal-iVal)*((mPos-iPos)/(jPos-iPos))
def getbandarea(S,W,below=False):
    intrp= np.array([S[0]+((S[-1]-S[0])*((W[i]-W[0])/(W[-1]-W[0]))) for i in range(len(W))])
    return np.sum(intrp-S) if not below else np.sum(np.where(intrp>S,intrp-S,0))
def getMAD(A):
    med=np.median(A)
    dev=A-med
    absdev=np.abs(dev)
    return np.median(absdev)

def getDiversePositions(sourceIF,sourceWV,targetWV,positionsCount=300,minimumDeepBands=3,feature='bandarea'):
    libSpectras=[]
    for m in range(len(sourceIF)):
        pixelInterPolFunc1=interp1d(sourceWV[m],sourceIF[m], kind='linear')
        prepSpectra=getStandardScaledSpectra(getContinuumRemovedSpectra_CH(removeSpikes(savitzky_golay(pixelInterPolFunc1(targetWV))))) #
        libSpectras.append(prepSpectra)
        
    D=[]
    for l in range(len(libSpectras)):
        D.extend(getAbsorptionPositions(libSpectras[l],minimumDeepBands=minimumDeepBands))
    
    diffList=[]
    for d in D:
        thisMdiffs=[]
        for k in range(len(libSpectras)):
            if feature=='banddepth':
                thisMdiffs.append(getbanddepth(libSpectras[k][d[0]],libSpectras[k][d[1]],libSpectras[k][d[2]],d[0],d[1],d[2]))
            elif feature=='bandarea':
                thisMdiffs.append(getbandarea(libSpectras[k][d[0]:d[2]],targetWV[d[0]:d[2]]))
            elif feature=='bandareaBlw':
                thisMdiffs.append(getbandarea(libSpectras[k][d[0]:d[2]],targetWV[d[0]:d[2]],below=True))
        thisMdiffs=np.array(thisMdiffs)
        thisrangeDiffScaled=(thisMdiffs-min(thisMdiffs))/(max(thisMdiffs)-min(thisMdiffs))
        diffList.append([getMAD(thisrangeDiffScaled)/np.median(thisrangeDiffScaled),d[0],d[1],d[2],d[3]])
    diffList.sort(key= lambda x:x[0],reverse=True)

    deepsFinal={}
    removeDeep=[]
    for d in range(len(diffList)):
        checkOverlap=False
        for i in range(diffList[d][1]-3,diffList[d][1]+4):
            for m in range(diffList[d][2]-2,diffList[d][2]+3):
                for j in range(diffList[d][3]-3,diffList[d][3]+4):
                    if deepsFinal.get((i,m,j)) is not None: 
                        checkOverlap=True
                        break
        if checkOverlap:
            removeDeep.append(d)
        else:
            deepsFinal[(diffList[d][1],diffList[d][2],diffList[d][3])]='dummy'
    removeDeep.reverse()
    for r in removeDeep:
        diffList.pop(r)
        
    return [diffList[d][1:-1] for d in range(min(len(diffList),positionsCount))]
#==========================================================
def getFeature(d,targetWV,rPos,feature='banddepth'):
    depth=np.array([np.max(d)-p for p in d])
    totalArea=np.sum(depth)
    maxDepth=np.max(depth)
    dd=[]
    for r in rPos:
        if 'Blw' in feature:
            thisarea=getbandarea(d[r[0]:r[2]],targetWV[r[0]:r[2]],below=True)
            if feature=='bandareaBlw':  dd.append(  thisarea )
            if feature=='areaBlwProportion': dd.append(  thisarea/totalArea )
        elif 'area' in feature:
            thisarea=getbandarea(d[r[0]:r[2]],targetWV[r[0]:r[2]])
            if feature=='bandarea':  dd.append(  thisarea )
            if feature=='areaProportion': dd.append(  thisarea/totalArea )
        elif 'depth' in feature:
            thisdepth=getbanddepth(d[r[0]],d[r[1]],d[r[2]],r[0],r[1],r[2])
            if feature=='banddepth':  dd.append(  thisdepth )
            if feature=='depthProportion': dd.append(  thisdepth/maxDepth )
    return np.array(dd)
def getFeatureData(D,targetWV,rPos,feature='banddepth'):
    D_r=[]
    for d in range(D.shape[0]):
        D_r.append(getFeature(D[d],targetWV,rPos,feature=feature))
    return np.array(D_r)
#==========================================================
def getMinMaxScaledSpectra(spectra):
    return (spectra-np.min(spectra))/(np.max(spectra)-np.min(spectra))
def getMinMaxScaledData(D):
    return np.array([getMinMaxScaledSpectra(d) for d in D])