#torch 1.7.1

import torch
from torch import nn



class GeneralFourier2d(torch.nn.Module):
    def __init__(self, c, h, w, log=False):
        super().__init__()

        self.log = log

        self.register_parameter(name='W1', param=torch.nn.Parameter(torch.ones (1, c, h, w // 2 + 1, 1)))
        self.register_parameter(name='B1', param=torch.nn.Parameter(torch.zeros(1, c, h, w // 2 + 1, 1)))
        self.register_parameter(name='W2', param=torch.nn.Parameter(torch.ones (1, c, h, w // 2 + 1, 1)))
        self.register_parameter(name='B2', param=torch.nn.Parameter(torch.zeros(1, c, h, w // 2 + 1, 1)))

        self.activation = nn.ReLU()

    def forward(self, x):
        w1 = self.activation(self.W1)
        w2 = self.activation(self.W2)
        b1 = self.activation(self.B1)
        b2 = self.activation(self.B2)

        rft_x = torch.rfft(x, signal_ndim=3, normalized=True, onesided=True)
        init_spectrum = (rft_x ** 2).sum(dim=-1, keepdim=True).sqrt()
        
        if self.log:
            spectrum = w2 * self.activation(w1 * (1 + init_spectrum).log() + b1) + b2
        else:
            spectrum = w2 * self.activation(w1 * init_spectrum + b1) + b2

        irf = torch.irfft(rft_x * spectrum / (init_spectrum + 1e-16),
                          signal_ndim=3, normalized=True, onesided=True, signal_sizes=x.shape[1:])

        return irf
    
class FilterSegModel(nn.Module):
    def __init__(self, model, c, h, w, log=False, filter_type='general2d'):
        super().__init__()
        self.model = model
        self.device = next(iter(model.parameters())).device
        if filter_type.lower() == 'general2d':
            self.filter = GeneralFourier2d(c, h, w, log).to(self.device)
#         elif filter_type.lower() == 'linear2d':
#             self.filter = LinearFourier2d(c, h, w, log).to(self.device)
        else:
            raise ValueError('Unknown filter type (possible values: general2d)')#, linear2d)')
            
    def forward(self, x):
        x = self.filter(x)
        x = self.model(x)
        
        return x