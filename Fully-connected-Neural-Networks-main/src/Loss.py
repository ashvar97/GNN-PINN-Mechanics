import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        loss=1

    def forward(self,prediction_tensor, label_tensor):
        self.prediction_tensor=prediction_tensor
        fwd_loss=-np.sum(np.where(label_tensor==1,np.log(prediction_tensor+np.finfo(prediction_tensor.dtype).eps),0))
        return fwd_loss


    def backward(self,label_tensor):
        bwd_loss= (-label_tensor/(self.prediction_tensor+ np.finfo(self.prediction_tensor.dtype).eps))
        return bwd_loss
