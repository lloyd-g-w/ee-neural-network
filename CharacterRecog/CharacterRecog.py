import cv2 
import numpy as np 
import matplotlib.pyplot as plt
import os
import pandas as pd
import random
import keyboard

trainers = pd.read_csv("emnist-balanced-train.csv", nrows=50000)
testers = pd.read_csv("emnist-balanced-test.csv", nrows=18000)





validTrainers = []
for i in range(50000):
    label = np.array(trainers.loc[i, "45"]).item() 
    if label < 36:
        validTrainers.append(i)

validTesters = []
for i in range(18000):
    label = np.array(testers.loc[i, "41"]).item() 
    if label < 36:
        validTesters.append(i)


files = ["hidden1bias.npy", "hidden2bias.npy", "outputbias.npy", "hidden1weights.npy", "hidden2weights.npy", "outputweights.npy"]


labelDictionary = {
    0: '0',
    1: '1',
    2: '2',
    3: '3',
    4: '4',
    5: '5',
    6: '6',
    7: '7',
    8: '8',
    9: '9',
    10: 'A',
    11: 'B',
    12: 'C',
    13: 'D',
    14: 'E',
    15: 'F',
    16: 'G',
    17: 'H',
    18: 'I',
    19: 'J',
    20: 'K',
    21: 'L',
    22: 'M',
    23: 'N',
    24: 'O',
    25: 'P',
    26: 'Q',
    27: 'R',
    28: 'S',
    29: 'T',
    30: 'U',
    31: 'V',
    32: 'W',
    33: 'X',
    34: 'Y',
    35: 'Z',
}



def loadnthCar(folder, n): #This function loads the nth image in a folder and returns the image and makes them binary
    filename = os.listdir(folder)[n]
    image = cv2.imread(os.path.join(folder, filename))
    return cv2.cvtColor(cv2.resize(image, (800, 500)), cv2.COLOR_BGR2GRAY)

def NPValues(i):
    with open("data.txt", "r") as f:
        j = f.read().split("\n\n")
        x = j[i].split("\n")	
        z = []
        for a in x:
            for b in a.split(" "):
                z.append(int(b))

        yhat = np.array(z).reshape((4,1))
        f.close()
    return (yhat)

def loadnthNChar(coords, imgNum, char):
    NP = cv2.resize(loadnthCar("Cars_1999", imgNum)[int(coords[1]):int(coords[3]), int(coords[0]):int(coords[2])], (196,28))
    return NP[0:28, 28*char:28*(char+1)]


def loadnthimage(imgNum):
    img = np.array(trainers.loc[imgNum, "0":]).reshape(28, 28)
    img = np.fliplr(img)
    img = np.rot90(img)
    label = np.array(trainers.loc[imgNum, "45"]).item()
    return img, label



def loadnthtester(imgNum):
    img = np.array(testers.loc[imgNum, "0":]).reshape(28, 28)
    img = np.fliplr(img)
    img = np.rot90(img)
    label = np.array(testers.loc[imgNum, "41"]).item()
    return img, label

def initalizeFiles():
    arrays = []
    
    hidden1bias = np.random.randn(40, 1)
    arrays.append(hidden1bias)
    
    hidden2bias = np.random.randn(40, 1)
    arrays.append(hidden2bias)
    
    outputbias = np.random.randn(36, 1)
    arrays.append(outputbias)
    
    hidden1weights = np.random.randn(40, 784)
    arrays.append(hidden1weights)
    
    hidden2weights = np.random.randn(40, 40)
    arrays.append(hidden2weights)
    
    outputweights = np.random.randn(36, 40)
    arrays.append(outputweights)
    
    for count, file in enumerate(files):
        np.save(file, arrays[count])
        
    return -1

def Softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=0)

def Sigmoid(x):
    return 1/(1 + np.exp(-x))


def SigmoidPrime(x):
    return Sigmoid(x)*(1 - Sigmoid(x))    

def forwardProp(w1, w2, w3, b1, b2, b3, input):
    z1 = np.dot(w1, input) + b1
    a1 = Sigmoid(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = Sigmoid(z2)
    z3 = np.dot(w3, a2) + b3
    a3 = Sigmoid(z3)
    a3 = Softmax(a3)
    return z1, a1, z2, a2, z3, a3



def weightDerivatives(delta, a): #This function calculates the weight derivatives where a is the activation of the previous layer and delta is the delta of the current layer
    return np.dot(delta, a.T)

def backProp(z1, a1, z2, a2, z3, a3, yhat, input, hidden2weights, outputweights):
    delC = 2*(a3 - yhat)
    outputDelta = np.multiply(delC, SigmoidPrime(z3))
    hidden2Delta = np.multiply(np.dot(outputweights.T, outputDelta), SigmoidPrime(z2))
    hidden1Delta  = np.multiply(np.dot(hidden2weights.T, hidden2Delta), SigmoidPrime(z1))
    
    dw1 = weightDerivatives(hidden1Delta, input)
    dw2 = weightDerivatives(hidden2Delta, a1)
    dw3 = weightDerivatives(outputDelta, a2)
    
    db1 = hidden1Delta
    db2 = hidden2Delta
    db3 = outputDelta
    
    return dw1, dw2, dw3, db1, db2, db3
    
def updateVals(w1, w2, w3, b1, b2, b3, dw1, dw2, dw3, db1, db2, db3, alpha):
    w1 -= alpha * dw1
    w2 -= alpha * dw2
    w3 -= alpha * dw3
    b1 -= alpha * db1
    b2 -= alpha * db2
    b3 -= alpha * db3
    
    np.save("hidden1weights.npy", w1)
    np.save("hidden2weights.npy", w2)
    np.save("outputweights.npy", w3)
    np.save("hidden1bias.npy", b1)
    np.save("hidden2bias.npy", b2)
    np.save("outputbias.npy", b3)
    
    return w1, w2, w3, b1, b2, b3
    
def avgThenUpdateDerviatives(derivativesList, w1, w2, w3, b1, b2, b3, alpha):
    total = [sum(x) for x in zip(*derivativesList)]
    avg = list(map(lambda x: x/len(derivativesList), total))
    w1, w2, w3, b1, b2, b3 = updateVals(w1, w2, w3, b1, b2, b3, avg[0], avg[1], avg[2], avg[3], avg[4], avg[5], alpha)
    return w1, w2, w3, b1, b2, b3
        

    
    
def main():
    k = 1
    while True:
        if keyboard.is_pressed("q"):
            quit()
        alpha = 0.0001
        w1 = np.load("hidden1weights.npy")
        w2 = np.load("hidden2weights.npy")
        w3 = np.load("outputweights.npy")
        b1 = np.load("hidden1bias.npy")
        b2 = np.load("hidden2bias.npy")
        b3 = np.load("outputbias.npy")
        derivatives = []
        for j in range(10000):
            imgNum = validTrainers[random.randint(0, len(validTrainers)-1)]
            image, desired1val = loadnthimage(imgNum)
            image = image.reshape(784, 1)
            z1, a1, z2, a2, z3, a3 = forwardProp(w1, w2, w3, b1, b2, b3, image)
            yhat = np.zeros(a3.shape)
            yhat[desired1val] = 1
            dw1, dw2, dw3, db1, db2, db3 = backProp(z1, a1, z2, a2, z3, a3, yhat, image, w2, w3)
            derivatives.append([dw1, dw2, dw3, db1, db2, db3])
            #print(a3)
            #print(z3)
        avgThenUpdateDerviatives(derivatives, w1, w2, w3, b1, b2, b3, alpha)
        print("Completed iteration: " + str(k))
        k+=1
        
def mainNoAVG():
    w1 = np.load("hidden1weights.npy")
    w2 = np.load("hidden2weights.npy")
    w3 = np.load("outputweights.npy")
    b1 = np.load("hidden1bias.npy")
    b2 = np.load("hidden2bias.npy")
    b3 = np.load("outputbias.npy")
    alpha = 0.00001
    k=0
    while True:
        if keyboard.is_pressed("q"):
            quit()
        derivatives = []
        imgNum = validTrainers[random.randint(0, len(validTrainers)-1)]
        image, desired1val = loadnthimage(imgNum)
        image = image.reshape(784, 1)
        z1, a1, z2, a2, z3, a3 = forwardProp(w1, w2, w3, b1, b2, b3, image)
        yhat = np.zeros(a3.shape)
        yhat[desired1val] = 1
        dw1, dw2, dw3, db1, db2, db3 = backProp(z1, a1, z2, a2, z3, a3, yhat, image, w2, w3)
        derivatives.append([dw1, dw2, dw3, db1, db2, db3])
        w1, w2, w3, b1, b2, b3 = avgThenUpdateDerviatives(derivatives, w1, w2, w3, b1, b2, b3, alpha)
        
        #print(a3)
        #print(yhat)
        #prediction = labelDictionary[np.where(a3==np.max(a3))[0][0]]
        #print(f"Predicted Character: {prediction} \nActual Character: {labelDictionary[desired1val]}")
        
        if k % 10 == 0:
            print("Completed iteration: " + str(k))
        k+=1


       
def visualTest():
    for i in range(10):
        w1 = np.load("hidden1weights.npy")
        w2 = np.load("hidden2weights.npy")
        w3 = np.load("outputweights.npy")
        b1 = np.load("hidden1bias.npy")
        b2 = np.load("hidden2bias.npy")
        b3 = np.load("outputbias.npy")
        imgNum = random.randint(0, len(validTesters)-1)
        goodimage, yhat = loadnthtester(validTesters[imgNum])
        image = goodimage.reshape(784, 1)
        z1, a1, z2, a2, z3, a3 = forwardProp(w1, w2, w3, b1, b2, b3, image)
        prediction = labelDictionary[np.where(a3==np.max(a3))[0][0]]
        print(f"Predicted Character: {prediction} \nActual Character: {labelDictionary[yhat]}")
        cv2.imshow("image",goodimage.astype(np.uint8))
        cv2.waitKey(0)
        cv2.destroyAllWindows() 
    return -1

def accuracyTest():
    accuracy = 0
    for i in range(1000):
        w1 = np.load("hidden1weights.npy")
        w2 = np.load("hidden2weights.npy")
        w3 = np.load("outputweights.npy")
        b1 = np.load("hidden1bias.npy")
        b2 = np.load("hidden2bias.npy")
        b3 = np.load("outputbias.npy")
        image, yhat = loadnthtester(validTesters[i])
        image = image.reshape(784, 1)
        z1, a1, z2, a2, z3, a3 = forwardProp(w1, w2, w3, b1, b2, b3, image)
        if (np.where(a3==max(a3))[0][0] == yhat):
            accuracy += 1
        if (i % 50 == 0):
            print(f"Accuracy: {accuracy/50*100}")
            accuracy = 0
    return -1

def cost():
    cost = np.empty((36, 1))
    for i in range(1000):
        w1 = np.load("hidden1weights.npy")
        w2 = np.load("hidden2weights.npy")
        w3 = np.load("outputweights.npy")
        b1 = np.load("hidden1bias.npy")
        b2 = np.load("hidden2bias.npy")
        b3 = np.load("outputbias.npy")
        imgNum = random.randint(0, len(validTesters)-1)
        image, yhat = loadnthtester(validTesters[imgNum])
        image = image.reshape(784, 1)
        z1, a1, z2, a2, z3, a3 = forwardProp(w1, w2, w3, b1, b2, b3, image)
        cost += np.square(a3-yhat)
    print(1/1000*cost)





       
       
#initalizeFiles()
#main()
#mainNoAVG()
accuracyTest()
#visualTest()
#cost()