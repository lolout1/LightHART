import math
from torch.optim.lr_scheduler import LambdaLR

class WarmupCosineScheduler(LambdaLR):
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.step_update_called = 0
        super(WarmupCosineScheduler, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch
        )
    
    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        
        progress = float(step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        decayed = cosine_decay * (1 - self.min_lr) + self.min_lr
        return decayed
    
    def step_update(self, step_num):
        self.step_update_called = step_num
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
