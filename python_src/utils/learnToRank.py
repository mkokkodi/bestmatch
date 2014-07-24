
import itertools
import numpy as np
from scipy import stats
import pylab as pl
from sklearn import svm, linear_model, cross_validation
from pandas import DataFrame
from sklearn.metrics import roc_auc_score
import evaluate as eval
from sklearn.metrics import accuracy_score

path = "/Users/mkokkodi/Desktop/bigdata/bestmatch/pairwiseTransform/"
trainFile = "train10.csv"
#application,opening,client,hours,bill_rate,pay_rate,jobs,cosine,pmi,skills_innerproduct,
#feedback,exams_innerproduct,
#rehires,edu,experience,number_tests,newContractor,label

def createFile():
    print "Creating file..."
    trainDf = DataFrame.from_csv(path+trainFile, header=False, sep=',',index_col=False)
    
    
    openings =  np.unique(trainDf['opening'].values)
    
    fout = open(path+"transformed10.csv","w")
      
    fout.write("hours,bill_rate,pay_rate,jobs,cosine,pmi,skills_innerproduct,feedback,exams_innerproduct,rehires,edu,experience,number_tests,newContractor,label\n")    
    
    features = "hours,bill_rate,pay_rate,jobs,cosine,pmi,skills_innerproduct,feedback,exams_innerproduct,rehires,edu,experience,number_tests,newContractor".split(",")
    for opening in openings:
        openingApplications =  trainDf[trainDf['opening']==opening]
        comb = itertools.combinations(range(openingApplications.shape[0]), 2)
    
        for (i, j) in comb:
            newLabel = openingApplications['label'].values[i]  - openingApplications['label'].values[j]
            if(newLabel != 0):
                tmpStr = ""
                for feature in features:
                    diff = openingApplications[feature].values[i] -openingApplications[feature].values[j]
                    tmpStr += str(diff)+","
                tmpStr+=str(newLabel)
                fout.write(tmpStr+"\n");
                
    print "File created."         
    fout.close()

def runSvm():
    print "Running SVM!"
    trainDf = DataFrame.from_csv(path+"transformed10.csv", header=False, sep=',',index_col=False)
    trainX = trainDf.ix[:,0:14]
    trainY = trainDf['label']
    print "Data loaded"   
    #lr = linear_model.LogisticRegression(C=5, penalty ='l1', tol=0.0000001)
    #lr.fit(trainX,trainY)
    clf = svm.SVC(kernel='linear', C=.1)
    clf.fit(trainX, trainY)
    print "fitting completed"
    #coef = clf.coef_.ravel() / linalg.norm(clf.coef_)
    #print coef 
    
    testDf = DataFrame.from_csv(path+"test10.csv", header=False, sep=',',index_col=False)
    testX = testDf.ix[:,3:17]
    predictions = clf.predict_proba(testX.values)
    #predictions = lr.predict_proba(testX.values)
    
    testDf['probability'] = predictions[:,1]
  
    print "Svm Completed. "
    #print accuracy_score(testY.values,lr.predict(testX.values))
    return testDf


print "Starting..."

#createFile()
dfSVM = runSvm()
svmRes = eval.AccTopN(dfSVM,6,'probability',False)
print "SVM AUC",roc_auc_score(dfSVM['label'].values, dfSVM['probability'])

for n,acc in  svmRes.iteritems():
    print n,acc
  
