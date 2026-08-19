import numpy as np
import Base

class SoftMax(Base.BaseLayer):
    def __init__(self):
        super(SoftMax, self).__init__()
        self.prob=1

    def forward(self,input_tensor):
        expfac=np.exp(input_tensor-np.max(input_tensor))
        sum_exp = np.sum(expfac,axis=1).reshape((input_tensor.shape[0], 1))
        self.prob = expfac / sum_exp
        return self.prob
    def backward(self,error_tensor):
        error_prev = self.prob * (error_tensor - np.sum((error_tensor*self.prob), axis = 1).reshape((error_tensor.shape[0], 1)))
        return error_prev
