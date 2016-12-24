# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 04:29:39 2016

@author: R16
"""



from __future__ import print_function
import numpy as np
np.random.seed(1337)  
import os
import scipy.io as sio
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from time import time
import matplotlib.pyplot as plt
import pandas as pd
#import charts


#batch_size = 110
nb_classes = 7
#nb_epoch = 20 #20

#
dataset= np.loadtxt(open("dataset/data_train.csv","rb"),delimiter=",")
#dataset= np.loadtxt(open("dataset/data_trainred.csv","rb"),delimiter=",")#reduksi dimensi1
#trainmat=sio.loadmat('dataset/train.mat')
#dsamp=trainmat["train"]
#testmat=sio.loadmat('dataset/test.mat')
#dsampt=testmat["test"]
#training_dataset = dsampt
#testing_dataset = dsampt
#training_dataset = np.append(dsamp,dsampt,axis=0)
data_test=np.loadtxt(open("dataset/data_test.csv","rb"),delimiter=",")
data_test/=255

training_dataset = dataset
testing_dataset = dataset
#training_dataset = dataset[16000:]
#testing_dataset = dataset[16000:]
(X_train, y_train), (X_test, y_test) = (training_dataset[:,0:10],training_dataset[:,10]),(testing_dataset[:,0:10],testing_dataset[:,10])
#(X_train, y_train), (X_test, y_test) = (training_dataset[:,0:7],training_dataset[:,7]),(testing_dataset[:,0:7],testing_dataset[:,7])#reduksi
#

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')


X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# ubah kelas target ke matriks biner


Y_train = np_utils.to_categorical(y_train-1, nb_classes)
Y_test = np_utils.to_categorical(y_test-1, nb_classes)

model = Sequential()
model.add(Dense(10, input_shape=(10,)))#hidden 1 : input 10 output 20 w=200 b = 20
model.add(Activation('relu'))
#model.add(Dropout(0.2))#layer dropout
#model.add(Dense(5))#hidden 2 : input 20 output 30 w=600 b =30
#model.add(Activation('relu'))
#model.add(Dropout(0.2))
model.add(Dense(7))#fc : input 30 output 7 w=210 b = 7
model.add(Activation('softmax'))
#model.summary()

rms = RMSprop()
model.compile(loss='categorical_crossentropy', optimizer='Adam',metrics=["accuracy"])

#model.fit(X_train, Y_train,
#          batch_size=batch_size, nb_epoch=nb_epoch, verbose=2,
#          validation_data=(X_test, Y_test))







def dekodebinertofloat(d,lim,kromosom):
    nkrom=[]
    x=0
    for i in range(int(len(kromosom)/lim)):
        pkrom =kromosom[x:x+lim].tolist()
        pkrom="".join(map(str,pkrom))
        nkrom=nkrom+[int(pkrom)/d]
        x+=lim
    return nkrom

def mutasibiner(kromosom,pM):
    for gen in kromosom:
        if np.random.rand() <= pM:
            gen=1 if gen==0 else 1
    return kromosom
    

def mutasi(kromosom,pM):
    for gen in range(len(kromosom)):
        if np.random.rand() <= pM:
            kromosom[gen] = kromosom[gen]+np.random.uniform(-1,1)
    
    return kromosom

def crossover(p1,p2,pC):
    if np.random.rand() <= pC:
        tipot=np.random.randint(len(p1)-1)
        c1=np.append(p1[tipot:],p2[:tipot])
        c2=np.append(p1[:tipot],p2[tipot:])
    else:
        c1=p1
        c2=p2
    return c1,c2

def rwheel(pop, fit, num):
#    print(fit)
    total_fitness = float(sum(fit))
    rel_fitness = [f/total_fitness for f in fit]
    #generate probabilitas kemunculan tiap individu
    probs = [sum(rel_fitness[:i+1]) for i in range(len(rel_fitness))]
    # populasi baru
    new_pop = []
    for n in range(num):
        r = np.random.rand()
        for (i, individual) in enumerate(pop):
            if r <= probs[i]:
                new_pop.append(individual)
                break
#    print(len(new_pop))
    return new_pop
def randomspecial(nPop):
  return np.concatenate((np.random.uniform(-3,4,size=(nPop,100)),
                   np.random.uniform(-2,2,size=(nPop,10)),
                   np.random.uniform(-4,3,size=(nPop,70)),
                   np.random.uniform(-2,2,size=(nPop,7))),axis=1)
def setweight(model,kromosom):
    
    hl1=np.array(model.layers[0].get_weights()[0]).shape
    hl2=np.array(model.layers[2].get_weights()[0]).shape
#    hl3=np.array(model.layers[4].get_weights()[0]).shape
                 
    whl1=hl1[0]*hl1[1]
    whl2=hl2[0]*hl2[1]
#    whl3=hl3[0]*hl3[1]

    kromosom=kromosom.reshape(1,kromosom.shape[0])
    w = kromosom[0,0:whl1]; w.shape = (hl1[0], hl1[1]);
    b = kromosom[0,whl1:whl1+hl1[1]]; b.shape = (hl1[1],)
    model.layers[0].set_weights([w,b])#set weights and biases
    w = kromosom[0,whl1+hl1[1]:whl1+hl1[1]+whl2]; w.shape = (hl2[0], hl2[1])
    b = kromosom[0,whl1+hl1[1]+whl2:whl1+hl1[1]+whl2+hl2[1]]; b.shape = (hl2[1],)
    model.layers[2].set_weights([w,b])
#    w = kromosom[0,whl1+hl1[1]+whl2+hl2[1]:whl1+hl1[1]+whl2+hl2[1]+whl3]; w.shape= (hl3[0], hl3[1])
#    b = kromosom[0,whl1+hl1[1]+whl2+hl2[1]+whl3:whl1+hl1[1]+whl2+hl2[1]+whl3+hl3[1]]; b.shape = (hl3[1],)
#    model.layers[4].set_weights([w,b])
    
    return model

def fitness(model,X_test,Y_test,kromosom):
    
    # fitness = 1/loss
    model=setweight(model,kromosom)
    score = model.evaluate(X_test,Y_test, verbose=0)
    return 1/score[0]

def fitnessbiner(model,X_test,Y_test,kromosom,d,lim):
    
    nkrom = np.array(dekodebinertofloat(d,lim,kromosom))
    model=setweight(model,nkrom)
    score = model.evaluate(X_test,Y_test, verbose=0)
    return 1/score[0]

def getelit(pop,fit):
    return np.array(fit).argsort()[-4:][::-1]
#    return np.array([pop[idx[0]],pop[idx[1]]])
#def getlowest(pop,fit):
def obsinitpop(nPop,nGen,x,y):
  t = 10
  xij=np.array([])
  bfit=np.array([])
  a=np.arange(x*t,y*t+1)/t
  for i  in a:
    for j in a:
      pop=np.random.uniform(i,j,size=(nPop,nGen))
      mfit=np.max([fitness(model,X_train,Y_train,pop[i]) for i in range(nPop)])
      bfit=np.append(bfit,mfit)
      xij=np.append(xij,(str(i)+""+str(j)))
  return bfit,xij
  
def run(model,maxep):
    #parameter GA
    nGen = model.count_params()#w+b
    nPop = 50
    pm=0.50
    pc=0.80
    ep=0
    
#    lim=11 # untuk biner
#    d=10000 # untuk biner
#    pop=np.random.rand(nPop,nGen)#initpop real
    pop=np.random.uniform(-1.5,0.9,size=(nPop,nGen))
    
#    pop=randomspecial(nPop)
#    pop=np.random.uniform(-0.4,0.4,size=(nPop,nGen))
#    pop=np.random.randint(2,size=(nPop,nGen))#initpop biner
#    print (len(pop[0]))
    lastfitness=0
    bestfitness=0
    elite=[]
    xfit=[]
    while ep < maxep:
        stime=time()
        if lastfitness != 0 and bestfitness !=0 and  lastfitness==bestfitness:
#        if count>=2:
#            pop=np.append(elite,np.random.uniform(-2,2,size=(nPop-4,nGen)),axis=0)
            pop=np.append(elite,randomspecial(nPop-4),axis=0)
            
#            count==0
        print("generasi",ep+1)
        #evalfit
#        fit= [fitness(model,X_test,Y_test,pop[i]) for i in range(nPop)]#integer
        fit= [fitness(model,X_train,Y_train,pop[i]) for i in range(nPop)]
#        fit= [fitness(model,X_test,Y_test,dekodebinertofloat(d,lim,pop[i])) for i in range(nPop)]#biner
        
        #elite
        idx=getelit(pop,fit)
        elite=np.array([pop[idx[i]] for i in range(4)])
        
        #slekesi ortu
        matpool=rwheel(pop,fit,nPop)

        pop=np.zeros((1,nGen))
        #crossover

        for c in range(int(nPop/2)):
            childs=crossover(matpool[2*c],matpool[2*c+1],pc)
            for c in childs:
                c=c.reshape(1,nGen)
                pop=np.append(pop,c,axis=0)
        pop = np.delete(pop,0,axis=0)
        
        #mutasi
        for i in pop:
            i=mutasi(i,pm)#integer
#            i=mutasibiner(i,pm)#biner

        #evalfit untuk survivor
#        fit= [fitness(model,X_test,Y_test,pop[i]) for i in range(nPop)]
        fit= [fitness(model,X_train,Y_train,pop[i]) for i in range(nPop)]

        idx= np.array(fit).argsort()[:4][::-1]#keluarkan 2 individu terburuk dari populasi
        pop = np.delete(pop,idx,axis=0)
        
        #masukkan 2kromosom elit ke populasi akhir
        for e in elite:
            e=e.reshape(1,nGen)    
            pop = np.append(pop,e,axis=0)
#        fit= [fitness(model,X_test,Y_test,pop[i]) for i in range(nPop)]
        fit= [fitness(model,X_train,Y_train,pop[i]) for i in range(nPop)]

        #ambilkromosom terbaik pada populasi akhir
        idx=getelit(pop,fit)
        elite=np.array([pop[idx[i]] for i in range(4)])
        bestchrom =elite[0]
        bestfitness=fit[idx[0]]
#        if lastfitness==bestfitness : 
#          count+=1         
        lastfitness=bestfitness
        
        print("fitnessnya - ",bestfitness)

        print('time elapsed :',time()-stime,' s')
        xfit=xfit+[bestfitness]
        ep+=1
    return bestchrom,xfit

def plot(model,X_train,Y_train):
  c = np.arange(7)
  y=np.array([(np.argmax(Y_train,axis=1)==i).sum() for i in c])
  pred =np.array([(np.argmax( model.predict(X_train),axis=1)==i).sum() for  i in c])
  fpred = np.abs(y-pred)
  pred=np.abs(pred-fpred)
  pred=pred.reshape(7,1)
  fpred=fpred.reshape(7,1)
  df=pd.DataFrame(np.append(pred,fpred,axis=1))
  ax=df.plot.bar(stacked=True)
  for p in ax.patches:
    ax.annotate(str(p.get_height()), xy=(p.get_x(), p.get_height()))
  print (df)


if __name__ == "__main__":
  '''uncomment ini jika ingin melatih NN dengan GA untuk optimasi W'''
#    maxep=3000
#    bestchrom,xfit=run(model,maxep)
#    model=setweight(model,bestchrom)
    #load w yng sudah pernah disimpan
#    plt.plot([i for i in range(maxep)],xfit)
#    plt.title("plot fitness terhadap epo")
#    plt.ylabel("fitness")
#    plt.xlabel("epo")
#    model.save_weights(os.path.join('%s_weights.h5' % model.name))
#    x,y=obsinitpop(50,187,-2,2)
  '''uncomment ini jika inign mengevaluasi NN  dengan W yang sudah diperoleh dari hasil latih 
  dengan data testing/validasi'''
  model.load_weights("saved_weights\sequential_23_weights.h5")#78%
#    model.load_weights("saved_weights\sequential_21_weights.h5")#82%
#    model.load_weights("saved_weights\sequential_10_weights.h5")#88%
#    model.load_weights("saved_weights\sequential_52_weights.h5")#97,2%
#    model.load_weights("saved_weights\sequential_62_weights.h5")#97,7%
  print('==========================================')#evaluate terhadap data testing
  score_ = model.evaluate(X_train,Y_train, verbose=0)
  print('Loss score:',score_[0],'Test accuracy:',score_[1])
  print('\n')
  
  pred=model.predict(X_train)#prediksi target
  plot(model,X_train,Y_train)