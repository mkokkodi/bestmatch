



#TO create the eval files  i need to:
#Have created the necessary training files
#1. Run the bayesian network (LearnNetwork.java)
#2. Run the footrule (FootruleBaseline)
from pandas import DataFrame
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import numpy as np

from sklearn import linear_model

intsToCats = {}
intsToCats['10']='Web Dev'
intsToCats['20']='Soft Dev'
intsToCats['40']='Writing'
intsToCats['60']='Des & Mult'
#category = "40"
resultPath = "/Users/mkokkodi/Dropbox/projects/bestmatch/data/results/"
path = "/Users/mkokkodi/Desktop/bigdata/bestmatch/"
#clientsMoreThan3.csv is not normalized train_single10_max40

def runLogistic(testFile, trainFile):
    #print "running logistic"
    lr = linear_model.LogisticRegression(C=5, penalty ='l1', tol=0.0000001)
    
    testDf = DataFrame.from_csv(path+testFile, header=False, sep=',',index_col=False)
    #print "test data loaded"
    trainDf = DataFrame.from_csv(path+trainFile, header=False, sep=',',index_col=False)
    #print "train data loaded"
    #selects columns 0 to 12
    testX = testDf.ix[:,3:18]
    testY = testDf['label']
    trainX = trainDf.ix[:,3:18]
    trainY = trainDf['label']
    lr.fit(trainX.values, trainY.values)

    predictions = lr.predict_proba(testX.values)
    #print accuracy_score(testY.values,lr.predict(testX.values))
    #print roc_auc_score(testY.values, predictions[:,1])
    #print confusion_matrix(testY.values, lr.predict(testX.values))

    testDf['probability'] = predictions[:,1]
    return testDf




    
def AccTopN(df, maxTopN, var, ascendingFlag):
    true = {}
    total = {}
    #print "running  at top n:",var
    openings =  np.unique(df['opening'].values)
    noOpenings = 0
    
    for opening in openings:
        #print "*************************************************"
        noOpenings +=1
        openingApplications =  df[df['opening']==opening]
        
        if len(openingApplications.index) < maxTopN:
            continue
        
        sortedDf = openingApplications.sort([var], ascending=ascendingFlag)
        n = 1
        for ind,row in sortedDf.iterrows():
            #print row['application'],row[var], row['label']
            
            if(n > maxTopN):
                break
            if n not in total:
                total[n] = 0
            if(row['label']==1):
                for topn in range(n,maxTopN):
                    if topn not in true:
                        true[topn] = 0
                    if topn not in total:
                        total[topn]=0
                    true[topn]  += 1
                    total[topn] +=1
                break
            total[n]+=1
            n+=1
        #break;
    #print "total Number of openings:",noOpenings
    #print "top 1:",true[1]
    res = {}       
    for n,nom in true.iteritems():
        res[n] = float(nom)/float(total[n])
    return res





def computeLift(modelDf,  variableToSortModel,modelName,outFile,category):
    sortedModel = modelDf.sort([variableToSortModel], ascending=False)
    
    totalRows = sortedModel.shape[0]
    step = (float(totalRows)/float(100.0))
    fiveStep = 5 * step
    acrossSet = float(sortedModel[sortedModel['label']==1].shape[0])/float(totalRows)
    topPercent = 1;
    i=0
    j=0;
    trueLabel=0
    for ind,row in sortedModel.iterrows():
        i +=1
        j+=1
        if(row['label']==1):
            trueLabel +=1
        if((i>step and topPercent == 1) or (i > fiveStep)):
            i=0
            lift = (float(trueLabel)/float(j))/float(acrossSet)
            #print topPercent,"\t",(float(trueLabel)/float(j))/float(acrossSet)
            outFile.write(modelName+","+intsToCats[category]+","+str(topPercent)+","+str(lift)+"\n")
            topPercent+=5
    #print row['probability'],"\t",row['label']
        
def runEval():
    bnPredictions = DataFrame.from_csv(path+'predictions/bnPredictions.csv', header=False, sep=',',index_col=False)
    footrulePreds = DataFrame.from_csv(path+'predictions/footrule.csv', header=False, sep=',',index_col=False)    
    accTopnF = open(resultPath+"accTopn.csv","w");
    accTopnF.write("model,category,topn,accuracy,baselineAcc,feedbackBaseline\n")
    liftsF = open(resultPath+"liftsComplete.csv","w");
    liftsF.write("model,category,top_prc,lift\n")
    
        
    for category in ("10","20","40","60"):
        print "****************** Category "+category+" *********************"
        #_normalized"
        testFile = "traintest/rankedInstancestest"+category+".csv"
        trainFile = "traintest/rankedInstancestrain"+category+".csv"
    
        dfLogistic = runLogistic(testFile, trainFile)
        
        feedbackDf = DataFrame.from_csv(path+"traintest/test"+category+".csv", header=False, sep=',',index_col=False)
    
    
        
        randomAtTopN =   AccTopN(feedbackDf,6,'application',False)
        baselineTopN = AccTopN(feedbackDf,6,'feedback',False)
         
        logistictAccTopN = AccTopN(dfLogistic,6,'probability',False)
    
       
        bnDF = bnPredictions[bnPredictions['category']==int(category)]
        bnAccTopN = AccTopN(bnDF,6,'probability',False)
    
        footrule = footrulePreds[footrulePreds['category']==int(category)]
    
        footruleTopN = AccTopN(footrule,6,'probability',False)
        
        computeLift(dfLogistic,'probability','Logit',liftsF,category)
        computeLift(bnDF,'probability','Bayesian Net',liftsF,category)
        computeLift(feedbackDf,'feedback','Feedback',liftsF,category)
        computeLift(footrule,'probability','Ranker Agg.',liftsF,category)
    
    
    
    #print accuracy_score(testY.values,lr.predict(testX.values))
        print "footrule AUC:",roc_auc_score(footrule['label'].values, footrule['probability'])
        print "BN AUC:",roc_auc_score(bnDF['label'].values, bnDF['probability'])
        print "Logistic AUC",roc_auc_score(dfLogistic['label'].values, dfLogistic['probability'])
        print "Feedback AUC",roc_auc_score(feedbackDf['label'].values, feedbackDf['feedback'])
    
        print "TopN \t Random  \t Feedback \t Logistic \t BayesNet \Ranker Aggregator"
        for n,acc in  logistictAccTopN.iteritems():
            print n,"\t",randomAtTopN[n],"\t",baselineTopN[n],"\t",acc,"\t",bnAccTopN[n],"\t",footruleTopN[n]
            accTopnF.write("Random,"+intsToCats[category]+","+str(n)+","+str(randomAtTopN[n])+","+str(randomAtTopN[n])+","+str(baselineTopN[n])+"\n")
            accTopnF.write("Feedback,"+intsToCats[category]+","+str(n)+","+str(baselineTopN[n])+","+str(randomAtTopN[n])+","+str(baselineTopN[n])+"\n")
            accTopnF.write("Logit,"+intsToCats[category]+","+str(n)+","+str(acc)+","+str(randomAtTopN[n])+","+str(baselineTopN[n])+"\n")
            accTopnF.write("Bayesian Net,"+intsToCats[category]+","+str(n)+","+str(bnAccTopN[n])+","+str(randomAtTopN[n])+","+str(baselineTopN[n])+"\n")
            accTopnF.write("Ranker Agg.,"+intsToCats[category]+","+str(n)+","+str(footruleTopN[n])+","+str(randomAtTopN[n])+","+str(baselineTopN[n])+"\n")
    accTopnF.close()     
    liftsF.close()     
        
        