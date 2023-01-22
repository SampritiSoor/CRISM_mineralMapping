import numpy as np
from CRISMrelatedMethods.dataRead import getVarInfo

mineralGroupNames=getVarInfo('./specifications.z','mineralGroupNames')
mineralGroupList=getVarInfo('./specifications.z','mineralGroupList')
mineralGroupDict=getVarInfo('./specifications.z','mineralGroupDict')
mineralIndexMap=getVarInfo('./specifications.z','mineralIndexMap')

def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

def getGroupPred(pred):        
    return np.array([mineralGroupDict[p] for p in pred])
def getFfoldFitScore(trainedModel,testDataX,testDataY,f=5,groupResult=True,foldScore=False):
    datasize=len(testDataX)
    datasizePerFold=int(datasize/f)
    startEnd=[(i*datasizePerFold,(i+1)*datasizePerFold) for i in range(f)]
    
    score=[]
    for itr in range(f):
        SE=startEnd[itr]
        Ppredict=trainedModel.predict(testDataX[SE[0]:SE[1]])
        if isinstance(Ppredict[0],np.int32):
            label=np.array([mineralIndexMap[Ppredict[p]] for p in range(Ppredict.shape[0])])
        else:
            label=np.array([mineralIndexMap[np.argmax(Ppredict[p])] for p in range(Ppredict.shape[0])])
        if groupResult: score.append(np.round((np.sum(np.where(getGroupPred(label)==getGroupPred(testDataY[SE[0]:SE[1]]),True,False))/datasizePerFold),4))
        else: score.append(np.round((np.sum(np.where(label==testDataY[SE[0]:SE[1]],True,False))/datasizePerFold),4))
    if foldScore:
        return score,np.round(np.mean(score),4),np.round(np.std(score),3)
    else:
        return np.round(np.mean(score),4),np.round(np.std(score),3)