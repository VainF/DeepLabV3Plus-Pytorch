from torch.optim.lr_scheduler import _LRScheduler, StepLR

class PolyLR(_LRScheduler):
    def __init__(self, optimizer, max_iters, power=0.9, last_epoch=-1):
        self.power = power
        self.max_iters = max_iters
        super(PolyLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        return [ base_lr * ( 1 - self.last_epoch/self.max_iters )**self.power
                for base_lr in self.base_lrs]