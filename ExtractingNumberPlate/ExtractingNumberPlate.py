import cv2 
import numpy as np 
from matplotlib import pyplot as pltpyt
import os
import keyboard

kernels = {
    "kernelblur" : np.array([[1/16, 1/8, 1/16], [1/8, 1/4, 1/8], [1/16, 1/8, 1/16]]),
    "kernelsobelx" : np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
    "kernelsobely" : np.array([[1, 2, 1], [0, 0 ,0], [-1, -2, -1]]),
    "smooth" : np.ones((5,5),np.float32)/25
}

files = ["hidden1bias.npy", "hidden2bias.npy", "outputbias.npy", "hidden1weights.npy", "hidden2weights.npy", "outputweights.npy"]


def loadnthimage(folder, n): #This function loads the nth image in a folder and returns the image and makes them binary
    filename = os.listdir(folder)[n]
    image = cv2.imread(os.path.join(folder, filename))
    return cv2.cvtColor(cv2.resize(image, (800, 500)), cv2.COLOR_BGR2GRAY)

def loadnthimageTest(folder, n): 
    filename = os.listdir(folder)[n]
    image = cv2.imread(os.path.join(folder, filename))
    return cv2.resize(image, (800, 500))

def initalizeFiles():
    arrays = []
    
    hidden1bias = np.random.randn(20, 1)
    arrays.append(hidden1bias)
    
    hidden2bias = np.random.randn(20, 1)
    arrays.append(hidden2bias)
    
    outputbias = np.random.randn(4, 1)
    arrays.append(outputbias)
    
    hidden1weights = np.random.randn(20, 400000)
    arrays.append(hidden1weights)
    
    hidden2weights = np.random.randn(20, 20)
    arrays.append(hidden2weights)
    
    outputweights = np.random.randn(4, 20)
    arrays.append(outputweights)
    
    for count, file in enumerate(files):
        np.save(file, arrays[count])
        
    return -1


def ReLU(x): 
    return np.maximum(0, x)

def ReLUPrime(x):
    return np.where(x > 0, 1, 0)
    
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
    a3 = ReLU(z3)
    return z1, a1, z2, a2, z3, a3


def actualValues(i):
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

def weightDerivatives(delta, a): #This function calculates the weight derivatives where a is the activation of the previous layer and delta is the delta of the current layer
    return np.dot(delta, a.T)

def backProp(z1, a1, z2, a2, z3, a3, yhat, input, hidden2weights, outputweights):
    delC = 2*(a3 - yhat)
    outputDelta = np.multiply(delC, ReLUPrime(z3))
    hidden2Delta = np.multiply(np.dot(outputweights.T, outputDelta), SigmoidPrime(z2))
    hidden1Delta  = np.multiply(np.dot(hidden2weights.T, hidden2Delta), SigmoidPrime(z1))
    
    hidden1weightsDerivative = weightDerivatives(hidden1Delta, input)
    hidden2weightsDerivative = weightDerivatives(hidden2Delta, a1)
    outputweightsDerivative = weightDerivatives(outputDelta, a2)
    
    hidden1biasDerivative = hidden1Delta
    hidden2biasDerivative = hidden2Delta
    outputbiasDerivative = outputDelta
    
    return hidden1weightsDerivative, hidden2weightsDerivative, outputweightsDerivative, hidden1biasDerivative, hidden2biasDerivative, outputbiasDerivative
    
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
    updateVals(w1, w2, w3, b1, b2, b3, avg[0], avg[1], avg[2], avg[3], avg[4], avg[5], alpha)

        

    
    
def main():
    k = 1
    while True:
        if keyboard.is_pressed("q"):
            quit()
        alpha = 0.0001
        for i in range(10):
            w1 = np.load("hidden1weights.npy")
            w2 = np.load("hidden2weights.npy")
            w3 = np.load("outputweights.npy")
            b1 = np.load("hidden1bias.npy")
            b2 = np.load("hidden2bias.npy")
            b3 = np.load("outputbias.npy")
            derivatives = []
            for j in range(10):
                imgNum = i*10 + j
                image = np.reshape(cv2.Canny(loadnthimage("Cars_1999", imgNum), 100, 200), (400000,1)) 
                yhat = actualValues(imgNum)
                z1, a1, z2, a2, z3, a3 = forwardProp(w1, w2, w3, b1, b2, b3, image)
                dw1, dw2, dw3, db1, db2, db3 = backProp(z1, a1, z2, a2, z3, a3, yhat, image, w2, w3)
                derivatives.append([dw1, dw2, dw3, db1, db2, db3])
            avgThenUpdateDerviatives(derivatives, w1, w2, w3, b1, b2, b3, alpha)
        print("Completed iteration: " + str(k))
        k+=1
       
def test():
    for i in range(10):
        w1 = np.load("hidden1weights.npy")
        w2 = np.load("hidden2weights.npy")
        w3 = np.load("outputweights.npy")
        b1 = np.load("hidden1bias.npy")
        b2 = np.load("hidden2bias.npy")
        b3 = np.load("outputbias.npy")
        imgNum = 101 + i
        image = np.reshape(loadnthimage("Cars_1999", imgNum), (400000,1)) 
        yhat = actualValues(imgNum)
        z1, a1, z2, a2, z3, a3 = forwardProp(w1, w2, w3, b1, b2, b3, image)
        print("Error: " + str(np.abs((np.sum(a3) - np.sum(yhat))/np.sum(yhat)*100)))
        cv2.imshow("image", cv2.resize(loadnthimageTest("Cars_1999", imgNum)[int(a3[1]):int(a3[3]), int(a3[0]):int(a3[2])], (196,28)))
        #cv2.imshow("image", cv2.resize(loadnthimageTest("Cars_1999", imgNum)[int(yhat[1]):int(yhat[3]), int(yhat[0]):int(yhat[2])], (196,28)))
        cv2.waitKey(0)
        cv2.destroyAllWindows() 


       
       
#initalizeFiles()
#main()
test()