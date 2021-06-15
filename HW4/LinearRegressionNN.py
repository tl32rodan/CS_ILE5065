import numpy as np
from collections import OrderedDict

# Template class of layers
class Layer(object):
    def __init__(self):
        pass
    
    def forward(self, x):
        pass
    
    def backward(self):
        pass

class Linear(Layer):
    def __init__(self, in_features: int, out_features: int, bias: bool=True) -> None:
        super(Layer,self).__init__()
        self.w = np.random.randn(in_features,out_features)
        self.has_bias = bias
        self.b = np.random.randn(out_features)
            
        # Initialize gradients to zeros --Seems like negligible--
        # self.grad = np.zeros(self.w.size)
        
        # Record input & output vector
        self.x = None
        self.out = None
        
    def forward(self, x):
        '''
            Input:
                x: np.array
        '''        
        self.x = x
        
        # Calculate output results
        if self.has_bias is True:
            # Output = XW + b
            self.out = np.dot(self.x,self.w) + self.b
        else:
            # Output = XW
            self.out = np.dot(self.x,self.w)
        
        return self.out
    
    def backward(self, prev_grad, lr=0.1):
        '''
            Input:
                prev_grad: np.array that comes from "next" layers
                lr : Learning rate
        '''
        # Calculate gradient of w, and update it with (learning rate*gradient)
        dw = np.dot(self.x.T,prev_grad)
        self.w -= lr * dw
        
        # Update bias if needed. Gradient of bias is the element-sum of prev_grad 
        if self.has_bias is True:
            db = np.sum(prev_grad,axis=0)
            self.b -= lr * db   
        
        # Return gradient that prpogate to next layer
        return np.dot(prev_grad,self.w.T)

    
class MSE(Layer):
    def __init__(self):
        self.y = None
        self.ground_truth = None
    
    def forward(self, y, ground_truth):
        self.y = y
        self.ground_truth = ground_truth
        v = np.mean((self.y-self.ground_truth)**2)
        if v > 100:
            print("v = ", v)
            print("y = ", self.y)
            print("ground_truth = ", self.ground_truth)
        return np.mean((self.y-self.ground_truth)**2)
    
    def backward(self,prev_grad=1,lr=0.1):
        '''
            prev_grad, lr: pseudo parameters
        '''
        return (2/self.y.shape[0])*(self.y-self.ground_truth)
class MAE(Layer):
    def __init__(self):
        self.y = None
        self.ground_truth = None
    
    def forward(self, y, ground_truth):
        self.y = y
        self.ground_truth = ground_truth
        return np.mean(self.y-self.ground_truth)
    
    def backward(self,prev_grad=1,lr=0.1):
        '''
            prev_grad, lr: pseudo parameters
        '''
        
        return (1/self.y.shape[0])*np.sign(self.y-self.ground_truth)

            
class LinearRegressionNN(object):
    def __init__(self, num_hidden_units=1, bias=True, loss_f='mse'):        
        loss_f_list = {'mse':MSE,
                      'mae':MAE}
        
        self.layers = OrderedDict()
        self.layers['hidden'] = Linear(1, num_hidden_units, bias=bias)
        self.layers['output'] = Linear(num_hidden_units, 1, bias=False)
        
        self.loss_func = loss_f_list[loss_f]()
        
    def forward(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
    
    def cal_loss(self, y, ground_truth):
        return self.loss_func.forward(y,ground_truth)
    
    def backward(self,lr=0.05):
        dy = self.loss_func.backward()
        
        # Reverse the layers list for easily conducting backward
        back_layers = list(self.layers.values())
        back_layers.reverse()
        for layer in back_layers:
            dy = layer.backward(dy,lr=lr)
