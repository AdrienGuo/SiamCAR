import torch

class SetCallback(object):
    def __init__(self):
        super(SetCallback, self).__init__()
    
    def on_train_begin(self):
        print('train begin')
        
    def on_train_end(self):
        print('train finished')
        
    def on_batch_end(self,status : dict) :
        print('batch:',status['batch'])
        print('train_loss:',status['loss'])
    
    def on_epoch_end(self,status: dict) :
        print('epoch:',status['epoch'])
        print('train_loss:',status['loss'])
        