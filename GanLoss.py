from torch import nn, optim
import torch


class loss_g(nn.Module):
    def __init__(self, gan_type='vanilla', real_label=1.0, fake_label=0.0):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(real_label))
        self.register_buffer('fake_label', torch.tensor(fake_label))
        if gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_type == 'lsgan':
            self.loss = nn.MSELoss()


    def get_labels(self, preds, target_label):
        if target_label:
            labels = self.real_label
        else:
            labels = self.fake_label
        return labels.expand_as(preds)


    def __call__(self, preds, target_label):
        labels = self.get_labels(preds, target_label)
        loss = self.loss(preds, labels)
        return loss