from pylab import rand,plot,show,norm
from math import exp, expm1
import numpy as np
import cPickle
import math


class MLNNMain:
 def __init__(self):
  """ perceptron initialization """
  self.w = rand(2) * 2 - 1 # weights
  self.learningRate = 0.01
  self.momentum = 0.8
  self.d = 3072
  self.k = 500 # number of hidden layers
  self.n = 10000
  self.nTest = 2000
  self.W1 = np.random.random_sample((self.d+1,self.k))# [[rand(1)*2-1 for x in range(self.k)] for x in range(self.d+1)] #for W10
  self.W2 = np.random.random_sample((self.k+1,1))#[[rand(1)*2-1 for x in range(1)] for x in range(self.k+1)]#for W20
  self.Z1 =  np.zeros((self.k+1,1))
  self.Z2 =  0
  self.f1 =  np.zeros((self.k+1,1))
  self.f2 =  0
  self.maxIteration = 10000
  self.globalError  = np.zeros((self.maxIteration,1))
  self.globalTestError = np.zeros((self.maxIteration,1))
  self.batchsize= 1
  self.dict = cPickle.load(open("cifar_2class_py2.p","rb"))
  self.shuffledData = np.random.permutation(self.n)
  self.D1 = np.zeros((self.d + 1, self.k))
  self.D2 = np.zeros((self.k + 1,1))
  self.G1 = np.zeros((self.d + 1, self.k))
  self.G2 = np.zeros((self.k + 1,1))
  self.dictTraindata= np.zeros((self.n, self.d))
  self.dictTrainLables= np.zeros((self.n, 1))
  self.dictTestdata= np.zeros((self.nTest, self.d))
  self.dictTestLables= np.zeros((self.nTest, 1))

 def accuracy_test(self):
     # check the test accuracy####################################################################
     iterTestError = 0
     for j in range(self.nTest) : # for each sample
        self.Z1 =np.append( np.dot(  np.append(self.dictTestdata.astype('float')[j],1),self.W1),1)
        self.f1 = np.vectorize(self.g)(self.Z1)

        self.Z2 = np.dot(self.f1 , self.W2)
        self.f2 = self.softmax(self.Z2)

        # if  ((1- self.f2) == 0):
        #     iterTestError += 0
        # else:
        #    iterTestError += self.dict['test_labels'][j] * math.log( self.f2)+ ( 1- self.dict['train_labels'][j]) * math.log(1- self.f2)  # desired response - actual response
        iterTestError  += abs(self.dict['test_labels'][j].astype('float') - self.f2)
     return iterTestError/self.nTest


 # This function if for computing the defrentiate of the RLUE function
 def g(self,a):
    if a > 0 :
      return a
    if a <= 0:
      return 0

 def softmax(self,a):
     return 1/( 1 + exp(-a))

 def gSubgradianet(self,a):
    if a > 0 :
      return 1
    if a <= 0:
      return 0


 def standardized(self,a,meanA,varA):
     return float((a-meanA)/varA)

# This function is for reading the input data and normalize the data and save it into the file
 def loadData(self):
     for i in self.dict:
       print i, self.dict[i].shape

     ##  standardized the training data#########
     import os.path
     if os.path.isfile('c:/untitled/MultiLayerNN/normalizedData.npy')==True:
       self.dictTraindata = np.load('c:/untitled/MultiLayerNN/normalizedData.npy')
       self.dictTestdata = np.load('c:/untitled/MultiLayerNN/normalizedTestData.npy')
     else:
         for i in range(self.d):
            meanA = np.mean(self.dict['train_data'][:,i].astype('float'))
            varA = np.var(self.dict['train_data'][:,i].astype('float'))
            if (varA > 0):
                self.dictTraindata[:,i] = (np.vectorize(self.standardized)(self.dict['train_data'][:,i].astype('float'),meanA,varA)).astype('float64')
                self.dictTestdata[:,i] = np.vectorize(self.standardized)(self.dict['test_data'][:,i].astype('float'),meanA,varA).astype('float64')
         np.save('c:/untitled/MultiLayerNN/normalizedData.npy',self.dictTraindata)
         np.save('c:/untitled/MultiLayerNN/normalizedTestData.npy', self.dictTestdata)


 def updateWeights(self,j):

        gg = np.dot(float(self.dict['train_labels'][self.shuffledData[j]])-self.f2, np.multiply(  np.vectorize( self.gSubgradianet)(self.Z1)  , np.transpose(self.W2)) )[0][0:self.k]
        self.G1 = np.add( self.G1 , np.dot( np.reshape(np.ravel(np.append( self.dictTraindata[self.shuffledData[j]],1)), (self.d+1,1)),np.reshape(np.ravel(gg),(1,self.k)) ))

        #        gg = np.dot(float(self.dict['train_labels'][self.shuffledData[j]])-self.f2, np.multiply(  np.vectorize( self.gSubgradianet)(self.Z1)  , np.transpose(self.W2)) )[0][0:self.k]

        #self.G1 = np.add( self.G1 , np.dot( np.reshape(np.ravel(np.append( self.dictTraindata.astype('float')[self.shuffledData[j]],1)), (self.d+1,1)),np.reshape(np.ravel(gg),(1,self.k)) ))
        if (j% self.batchsize == 0):
             self.D1 = np.subtract( np.multiply( self.momentum ,self.D1) , np.multiply( self.learningRate * 1/(self.batchsize) , self.G1) )
             self.W1 = np.add(self.W1 , self.D1)
             self.G1 = np.zeros((self.d + 1, self.k))

        self.G2 = np.add( self.G2 , np.reshape(np.ravel(  np.multiply (float(self.dict['train_labels'][self.shuffledData[j]])-self.f2 , self.f1 )), (self.k+1,1)))
        if (j% self.batchsize == 0):
             self.D2 = np.subtract( np.multiply( self.momentum ,self.D2) , np.multiply( self.learningRate * 1/(self.batchsize) , self.G2) )
             self.W2 = np.add(self.W2 , self.D2)
             self.G2 = np.zeros((self.k+1,1))


 ################################### Main Function #################################
 def train(self):

  learned = False

  # for the test data too
  self.loadData()
  iteration = 0
  iterationTest=0

  while not learned: # the loop over epoch
      #(5) Training Monitoring: For each epoch in training, this function  evaluate the training objective, testing objective,
      # training misclassification error rate (error is 1 for each example if misclassifies, 0 if correct), testing misclassification error rate
     self.globalError[iteration] = 0.0

     #uniformly sample train
     self.shuffledData = np.random.permutation(self.n)
     for j in range(self.n) : # for each sample

        ############################ compute variables ###########################
        self.Z1 =np.append( np.dot(  np.append(self.dictTraindata.astype('float')[self.shuffledData[j]],1),self.W1),1)
        self.f1 = np.vectorize(self.g)(self.Z1)

        self.Z2 = np.dot(self.f1 , self.W2)
        self.f2 = self.softmax(self.Z2)

        # if  ((1- self.f2) == 0):
        #     iterError=0
        # else:
        #    iterError = self.dict['train_labels'][self.shuffledData[j]] * math.log( self.f2)+ ( 1- self.dict['train_labels'][self.shuffledData[j]]) * math.log(1- self.f2)  # desired response - actual response
        iterError = self.dict['train_labels'][self.shuffledData[j]] - self.f2  # desired response - actual response
        self.globalError[iteration] += abs(iterError)
        print("\n"+ str(self.globalError[iteration]))

        ################## update weights ###########################
        self.updateWeights(j)
        if (j%1000==0):

         ################# compute test data accuracy ################
         self.globalTestError[iterationTest] = self.accuracy_test() #testing misclassification error rate
         print("\n"+ str(self.globalTestError[iterationTest]))
         iterationTest = iterationTest+1

     if self.globalTestError[iterationTest-1] == 0.0 or iteration >= self.maxIteration: # stop criteria
        print 'iterations',iteration
        learned = True # stop learning
     self.globalError[iteration] += abs(iterError)/ self.n #training misclassification error rate
     iteration += 1
  plot(range(iteration),self.globalTestError[0:iterationTest],'ob' )
  np.save("alldata", self)




mlnn= MLNNMain()
iterationError= mlnn.train()
