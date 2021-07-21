import torch
from torch import nn, optim
from discriminator import PatchDiscriminator
from GanLoss import loss_g


def init_weights(net, init='xavier', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and 'Conv' in classname:
            if init == 'norm':
                nn.init.normal_(m.weight.data, mean=0.0, std=gain)
            elif init == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)

            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif 'BatchNorm2d' in classname:
            nn.init.normal_(m.weight.data, 1., gain)
            nn.init.constant_(m.bias.data, 0.)

    net.apply(init_func)
    print(f"model initialized with {init} initialization")
    return net


def model_device(model):
    model = init_weights(model)
    return model


class MainModel(nn.Module):
    def __init__(self, gen=None, lr_G=1e-4, lr_D=1e-4,
                 beta1=0.5, beta2=0.999, lambda_L1=100.):
        super().__init__()
        self.lambda_L1 = lambda_L1
        self.gen = gen
        self.dis = model_device(PatchDiscriminator(input_c=3, n_down=3, num_filters=64))
        self.gantype = loss_g(gan_type='vanilla')
        self.L1criterion = nn.L1Loss()
        self.opt_G = optim.Adam(self.gen.parameters(), lr=lr_G, betas=(beta1, beta2))
        self.opt_D = optim.Adam(self.dis.parameters(), lr=lr_D, betas=(beta1, beta2))


    def grad(self, model, requires_grad=True):
        for p in model.parameters():
            p.requires_grad = requires_grad


    def channeling_input(self, data):
        self.L = data['L']
        self.ab = data['ab']


    def forward(self):
        self.fake_color = self.gen(self.L)


    def backward_D(self):
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.dis(fake_image.detach())
        self.loss_D_fake = self.gantype(fake_preds, False)
        real_image = torch.cat([self.L, self.ab], dim=1)
        real_preds = self.dis(real_image)
        self.loss_D_real = self.gantype(real_preds, True)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()


    def backward_G(self):
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.dis(fake_image)
        #self.loss_G_GAN = self.gantype(fake_preds, True)
        self.loss_G_L1 = self.L1criterion(self.fake_color, self.ab) * self.lambda_L1
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize(self):
        self.forward()
        self.dis.train()
        self.grad(self.dis, True)
        self.opt_D.zero_grad()
        self.backward_D()
        self.opt_D.step()

        self.gen.train()
        self.grad(self.dis, False)
        self.opt_G.zero_grad()
        self.backward_G()
        self.opt_G.step()
