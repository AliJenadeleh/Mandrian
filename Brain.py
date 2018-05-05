import numpy as np

class Brain:

    def IsMandarin(self,value):
        if value >= 0.5 :
            print("Mandrian")
        else : 
            print("Not Mandrian")
    
    def initData(self):
        self.unknow = (7,9,4)
        # Lines,Rectangles,Colours,Mandrian
        self.data = [(6,10,4,0),(4,8,5,0),(5,7,4,1),(5,8,4,1),(5,10,5,0),(6,8,6,1),(7,14,5,0)]
        self.dataLen = len(self.data)

    def __init__(self):
        self.initData()
        self.w1 = np.random.randn()
        self.w2 = np.random.randn()
        self.w3 = np.random.randn()
        self.bi = np.random.randn()
        self.alpha = 0.2
        self.trainingLoop = 50000
    
    def Sigmoid(self,z):
        Z = - z
        return 1 / (1 + np.exp(Z))
    

    def Cost(self,target,value):
        return (value - target) ** 2
    
    def DeCost(self,target,value):
        return 2 * (value - target)
    
    def Training(self):
        print("Trainig start")
        for i in range(self.trainingLoop):
            inx = np.random.choice(range(self.dataLen))
            item = self.data[inx]

            z = (self.w1 * item[0]) + (self.w2 * item[1]) + (self.w3 * item[2]) + self.bi
            pred = self.Sigmoid(z)
            cost = self.Cost(item[3],pred)

            dpred = self.Sigmoid(z) * (1 - self.Sigmoid(z))
            dcost = self.DeCost(item[3],pred)

            dtw1 = item[0]
            dtw2 = item[1]
            dtw3 = item[2]
            dtbi = 1

            dw1 = dpred * dcost * dtw1
            dw2 = dpred * dcost * dtw2
            dw3 = dpred * dcost * dtw3
            dbi = dpred * dcost * dtbi

            self.w1 -= self.alpha * dw1
            self.w2 -= self.alpha * dw2
            self.w3 -= self.alpha * dw3
            self.bi -= self.alpha * dbi
        print("W1 :",self.w1,"W2 :",self.w2,"W3 :",self.w3,"B :",self.bi)
        print("Training done.")
    #### Lines,Rectangles,Colours,Mandrian
    def WhatIs(self,lines,rects,colours):
        z = (self.w1 * lines) + (self.w2 * rects) + (self.w3 * colours) + self.bi
        pred = self.Sigmoid(z)
        self.IsMandarin(pred)

    def IsItMandrian(self):
        self.WhatIs(self.unknow[0],self.unknow[1],self.unknow[2])
