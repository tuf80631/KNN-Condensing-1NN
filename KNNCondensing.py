import numpy as np  
 import math  
 import random  
 from datetime import datetime  
 from random import randint  
 import pandas as pd  
 ##     Imports data for the feature values and labels of the training set, and the feature values of the testing set into npArrays  
 trainX = np.genfromtxt("Letter Recog 15000 Training Set.csv", dtype = 'int', delimiter=",", usecols=(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16))  
 trainY = np.genfromtxt("Letter Recog 15000 Training Set.csv", dtype = 'str', delimiter=",", usecols=(0))  
 testX = np.genfromtxt("Letter Recog 5000 Testing Set.txt", dtype = 'int', delimiter=",", usecols=(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16))  
 testY = np.genfromtxt("Letter Recog 5000 Testing Set.txt", dtype = 'str', delimiter=",", usecols=(0))  
 ## calculateED(i_arrayA, i_arrayB)  
 ## Purpose: To calculate the Eucledian Distance between two npArrays.   
 def calculateED(i_arrayA, i_arrayB):  
      i = 0;  
      sumOfSquares = 0;  
      while (i<=15): ## dimension of nparrays. 16 feature values for each example  
           sumOfSquares += (i_arrayA[i]-i_arrayB[i])*(i_arrayA[i]-i_arrayB[i])  
           i=i+1  
      euclideanDistance = math.sqrt(sumOfSquares)  
      return euclideanDistance  
 ##     testknn(i_trainX, i_trainY, i_testX, k)  
 ##     Purpose: To return predicted labels of test data (i_testX) using regular KNN algorithm  
 def testknn(i_trainX, i_trainY, i_testX, k):  
      o_testY = np.empty(len(i_testX),dtype='str') ## nparray for predicted testing labels  
      S = np.zeros(len(i_trainX))   ## nparray for sorting the ED between all the training set and a testing label  
      firstKsortedTrainY = np.empty(k,dtype='str') ##nparray to hold the labels of first k training exmaples closes to the selected testing example  
      i = 0  
      while (i < len(i_testX)): ## iterate through every testing set   
           j = 0  
           while (j <len(i_trainX)): ## calculate ED between selected testing example (index = i) and every training set data   
                S[j] = calculateED(i_trainX[j],i_testX[i])  
                j = j+1  
           inds = S.argsort()[:k] ## return the first k closes indices of training examples  
           m = 0  
           while (m<k):  
                firstKsortedTrainY[m]=(i_trainY[inds[m]]) ## return the first k training example labels that are closes to the testing example  
                m = m + 1  
           u, indices = np.unique(firstKsortedTrainY, return_inverse=True) ## find the most common label among first k training examples  
           mostFrequentNeighbor = u[np.argmax(np.bincount(indices))]  
           o_testY[i] = (mostFrequentNeighbor) ## assign most common label to the predicted npArray  
           i = i+1                 
      return o_testY  
 ## condensedata(i_trainX, i_trainY)  
 ## condenses the training set (i_trainX) using the 1NN condensing algorithm  
 def condensedata(i_trainX, i_trainY):  
      OriginalTrain = set()     #Set to hold indices of the Training Set  
      CondensedTrain = set() #Set to hold indices of the condensed set  
      anyMoreToAddToPrototype = True #Boolean value that will end the entire algorithm if there is no more to add from Training Set to Condensed Set  
      ##initiliaze index for original set  
      i = 0  
      while (i<len(i_trainX)):  
           OriginalTrain.add(i)  
           i=i+1  
      while (anyMoreToAddToPrototype): #The while loop will end if algorithm makes a pass through one iteration of the Original set and cannot find an example to add to Condensed Set  
           anyMoreToAddToPrototype = False  
           j = 0  
           while (j < len(OriginalTrain)): ## Make one iteration through every Training Set example  
                minCondensedSetIndex = 200000  
                minCondensedSetDistance = 200000  
                randomOriginalTrain = int(random.sample(OriginalTrain, 1)[0]) ##Find a random Training set example index  
                if (len(CondensedTrain)==0): #If the random training example is the first to be chosen, remove from training set and put into condensed set  
                     OriginalTrain.discard(randomOriginalTrain)  
                     CondensedTrain.add(randomOriginalTrain)  
                else:  
                     for item in CondensedTrain: #Find the condensed set example that is closest to the random training example by using ED  
                          euclideanDistance = calculateED(i_trainX[randomOriginalTrain], i_trainX[item])  
                          if (euclideanDistance<minCondensedSetDistance):  
                               minCondensedSetIndex = item  
                               minCondensedSetDistance = euclideanDistance  
                     #If the closest condensed set example's label is different from the random trianing example, remove from Original set and put into condensed set  
                     if (i_trainY[randomOriginalTrain]!=i_trainY[minCondensedSetIndex] and minCondensedSetIndex!=200000):   
                          OriginalTrain.discard(randomOriginalTrain)  
                          CondensedTrain.add(randomOriginalTrain)  
                          anyMoreToAddToPrototype = True #If this line is never hit after one iteration in the Original Set, the entire algorithm will end.   
                j+=1  
      condensedIdx = np.zeros(len(CondensedTrain), dtype = 'int')  
      h = 0  
      for item in CondensedTrain:  
           condensedIdx[h]=item  
           h+=1  
      print(len(CondensedTrain))  
      return condensedIdx # return the condensed training set label as a npArray vector.  
 ## classificationAccuracy(predictedY, actualY)  
 ## Purpose: To find how well the knn and condensed knn algorithms was able to predict the testing set labels.  
 def classificationAccuracy(predictedY, actualY):  
           total = len(actualY)  
           match = 0  
           i = 0  
           while (i<total):  
                if(predictedY[i]==actualY[i]):  
                     match+=1  
                i+=1  
           return match/total*100  
 def generateCondensedKnnReport(i_trainX, i_trainY, i_testX, i_testY, k, N):  
      fileTitle = "k="+str(k)+" condensed"+" N="+str(N)  
      print(fileTitle)  
      start_time = datetime.now()  
      PredictedTestY = testknn(i_trainX,i_trainY,i_testX,k)  
      print(classificationAccuracy(PredictedTestY, i_testY ))  
      np.savetxt(fileTitle,PredictedTestY, fmt="%s")  
      end_time = datetime.now()  
      print('Duration: {}'.format(end_time - start_time))  
      print(" ")  
 def generateKnnReport(i_trainX, i_trainY, i_testX, i_testY, k, N):  
      fileTitle = "k="+str(k)+" knn"+" N="+str(N)  
      print(fileTitle)  
      start_time = datetime.now()  
      PredictedTestY = testknn(i_trainX,i_trainY,i_testX,k)  
      print(classificationAccuracy(PredictedTestY, i_testY ))  
      np.savetxt(fileTitle,PredictedTestY, fmt="%s")  
      end_time = datetime.now()  
      print('Duration: {}'.format(end_time - start_time))  
      print(" ")  
      #y_actu = pd.Series(i_testY, name='Actual')  
      #y_pred = pd.Series(PredictedTestY, name='Predicted')  
      #df_confusion = pd.crosstab(y_actu, y_pred)  
      #print(df_confusion)  
      #df_confusion.to_csv('SVM_Confusion_Matrix.csv')  
 def classificationAccuracy(predictedY, actualY):  
           total = len(actualY)  
           match = 0  
           i = 0  
           while (i<total):  
                if(predictedY[i]==actualY[i]):  
                     match+=1  
                i+=1  
           return match/total*100  
 ##10000  
 randomSubSampleIndexSet = set()  
 subSampleTrainX = np.zeros((10000,16), dtype = 'int')  
 subSampleTrainY = np.empty(10000,dtype='str')  
 i = 0  
 while (i<10000):  
      randomSubSampleIndexSet.add(randint(0,14999))  
      i+=1  
 i=0  
 for item in randomSubSampleIndexSet:  
      b = 0  
      while (b<=15):  
           subSampleTrainX[i][b] = trainX[item][b]  
           b+=1  
      subSampleTrainY[i] = trainY[item]  
      i+=1  
 print("condense data N = 10000")  
 start_time = datetime.now()       
 condensedTrainIndexes= condensedata(subSampleTrainX, subSampleTrainY)  
 end_time = datetime.now()  
 print('Duration: {}'.format(end_time - start_time))  
 print(" ")  
 condensedTrainX = np.zeros((len(condensedTrainIndexes),16), dtype = 'int')  
 condensedTrainY = np.zeros(len(condensedTrainIndexes), dtype = 'str')  
 o = 0  
 for item in condensedTrainIndexes:  
      i = 0  
      while (i<=15):  
           condensedTrainX[o][i] = trainX[item][i]  
           i+=1  
      condensedTrainY[o] = trainY[item]  
      o+=1       
 kkk = 1  
 while (kkk<=9):  
      generateCondensedKnnReport(condensedTrainX, condensedTrainY, testX, testY, kkk, 10000)  
      kkk+=2  