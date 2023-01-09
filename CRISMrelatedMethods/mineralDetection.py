from CRISMrelatedMethods.preprocessing import *
from CRISMrelatedMethods.dataRead import getVarInfo
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler,MinMaxScaler
#=============================================================================================
def getPreprocessedData(wavelengths,imgdata,framedata,pipeline='cr_sm_CR_SS_FE',returnBandCroppedImage=False,LINES=None,LINE_SAMPLES=None,WR='bandarea',FE='banddepth'):
    spectralWavelengthSet=getVarInfo('specifications.z','spectralWavelengthSet')
    spectralIF=getVarInfo('specifications.z','spectralIF')
    spectralWavelength=getVarInfo('specifications.z','spectralWavelength')
    
    steps=pipeline.split('_')
    for step in steps:
        # domain collating
        if step=='cr':
            imgdata=np.array([interp1d(wavelengths,S, kind='linear')(spectralWavelengthSet) for S in imgdata])
            
            if returnBandCroppedImage and LINES is not None and LINE_SAMPLES is not None:
                try:
                    I_cr=imgdata.reshape(LINES,LINE_SAMPLES,len(spectralWavelengthSet))
                except:
                    I_cr=[]
            else:
                I_cr=[]

        # smoothing
        if step=='sm':
            imgdata=getSmoothedData(imgdata,prep='sm',mask=framedata)

        #continuum removal
        if step=='CR':
            imgdata=getContinuumRemovedData(imgdata,mask=framedata)

        # standardization
        if step=='SS':
            imgdata=getStandardScaledData(imgdata,mask=framedata)

        # feature extraction
        if step=='FE':
            P=getDiversePositions(spectralIF,spectralWavelength,spectralWavelengthSet,feature=WR)
            imgdata=getFeatureData(imgdata,spectralWavelengthSet,P,feature=FE)

            scaler = MinMaxScaler()
            scaler.fit(imgdata)
            imgdata=scaler.transform(imgdata)
            
    if returnBandCroppedImage:
        return imgdata,I_cr
    else:
        return imgdata
#=============================================================================================
from functools import reduce
def getHexColorStr(aColor):
    hexStr="#"+"00"*len(aColor)
    hexStrL=list(hexStr)
    for c in range(len(aColor)):
        hexStrL[1+(c+1)*2-len(hex(aColor[c])[2:]):1+(c+1)*2]=hex(aColor[c])[2:]
    return "".join(hexStrL)

def generateDominatingMineralPlots(result,I_cr,imageFrame,maximumMineralCount=3):
    spectralWavelengthSet=getVarInfo('specifications.z','spectralWavelengthSet')
    mineralColours=getVarInfo('specifications.z','mineralColours')
    mineralNames=getVarInfo('specifications.z','mineralNames')
    deepWavelengths=getVarInfo('specifications.z','deepWavelengths')
    spectralIF=getVarInfo('specifications.z','spectralIF')
    spectralWavelength=getVarInfo('specifications.z','spectralWavelength')
    resultColors={'ANN':'goldenrod','CNN':'goldenrod','RFC':'springgreen','SVC':'palevioletred'}

    detectedMinerals={}
    for R in result:
        if isinstance(result[R][0],np.int32): Pred=result[R].reshape(I_cr.shape[0],I_cr.shape[1])
        else: Pred=np.array([np.argmax(p) if np.max(p)>.67 else len(mineralNames) for p in result[R]]).reshape(I_cr.shape[0],I_cr.shape[1]) #

        imgPixels=(I_cr.shape[0]*I_cr.shape[1]-np.sum(imageFrame))
        detectedMinerals[R]={}
        for i in range(len(mineralNames)):
            thisDetected=np.where(Pred==i,True,False)&~imageFrame
            detectedRatio=np.sum(thisDetected)/imgPixels
            if detectedRatio>.03:
                mineralImage=np.full((LINES,LINE_SAMPLES,3),255)        
                detectedMinerals[R][i]={'mineralPix':thisDetected,'mineralName':mineralNames[i],'mineralColour':mineralColours[i],'detectedRatio':detectedRatio}
    
    commonDetectedIdx=list(reduce(lambda i, j: i & j, (set(x) for x in [list(detectedMinerals[R].keys()) for R in result] )))
    
    for R in result:
        L=list(detectedMinerals[R].keys())
        for i in L:
            if i not in commonDetectedIdx:
                detectedMinerals[R].pop(i)
                
    toRemove=[(np.min([detectedMinerals[R][c]['detectedRatio'] for R in result]),c) for c in commonDetectedIdx]
    toRemove.sort(key=lambda x: x[0],reverse=True)
    toRemove=toRemove[maximumMineralCount:]
    for t in toRemove:
        for R in result:
            detectedMinerals[R].pop(t[1])
        commonDetectedIdx.remove(t[1])
                
    for R in result:
        if len(detectedMinerals[R])==0: continue
        plt.figure(figsize=(2,.5))
        plt.text(0,0,R,fontsize=20,fontweight='bold')
        plt.axis("off")
        plt.show()
        h,w=min(int(30/len(detectedMinerals[R])),7)*2,min(int(30/len(detectedMinerals[R])),7)*len(detectedMinerals[R])        
        plt.figure(figsize=(w,h))
        subplotCount=0
        for i in detectedMinerals[R]:
            subplotCount+=1
            plt.subplot(1,len(detectedMinerals[R]),subplotCount)
            B=np.where(imageFrame,np.median(image[100,:,:]),image[100,:,:])
            B=np.where(imageFrame,np.max(B),B)
            B=(((B-np.min(B))/(np.max(B)-np.min(B)))*255).astype(np.int32)
            B_rgb=np.repeat(np.expand_dims(B,axis=2),3, axis=2)
            D=np.repeat(np.expand_dims(detectedMinerals[R][i]['mineralPix'],axis=2),3, axis=2)
            B_rgb=np.where(D==np.full(3,True),np.array([152, 245, 76]),B_rgb) #detectedMinerals[R][i]['mineralColour']
#             cv2.imwrite(detectedMinerals[R][i]['mineralName']+'.png',B_rgb)   '#98f54c'
            plt.imshow(B_rgb)
            plt.title(detectedMinerals[R][i]['mineralName'],fontsize=13,fontweight='bold',color='black') #getHexColorStr(detectedMinerals[R][i]['mineralColour'])
            plt.axis("off")
        plt.show()
        
    if len(commonDetectedIdx)>0:
        plt.figure(figsize=(len(commonDetectedIdx)*4,6))
        subplotCount=0
        for c in commonDetectedIdx:
            subplotCount+=1
            plt.subplot(1,len(commonDetectedIdx),subplotCount) 
            prevMin=0
            for R in result:
                detectedIdx=[(i,j) for i in range(LINES) for j in range(LINE_SAMPLES) if detectedMinerals[R][c]['mineralPix'][i,j]]
                randomIndices=random.sample(range(0, len(detectedIdx)), 500)
                S=np.zeros(len(spectralWavelengthSet))
                for s in randomIndices:
                    S+=I_cr[detectedIdx[s]]
                S/=500
                thisPlot=S+(prevMin-np.max(S))
                plt.plot(spectralWavelengthSet,thisPlot,color=resultColors[R])  
                plt.text(spectralWavelengthSet[0],thisPlot[0],R,fontsize=12,fontweight='bold') 
                prevMin=np.min(thisPlot)
            S=interp1d(spectralWavelength[c],spectralIF[c], kind='linear')(spectralWavelengthSet)
            thisPlot=S+(prevMin-np.max(S))
            plt.plot(spectralWavelengthSet,thisPlot,color='b')  
            plt.text(spectralWavelengthSet[0],thisPlot[0],'MICA',fontsize=12,fontweight='bold')
            plt.title(mineralNames[c],fontsize=20,fontweight='bold')  
            plt.yticks([])
            if deepWavelengths.get(c) is None:
                plt.xticks([])
            else:
                plt.xticks(deepWavelengths[c]['A'],deepWavelengths[c]['A_'])
            plt.grid(axis='x')
        plt.tight_layout(pad=2.0)
        plt.show()
#=============================================================================================s