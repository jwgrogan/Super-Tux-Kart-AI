import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms as T

def spatial_argmax(logit):
    """
    Compute the soft-argmax of a heatmap
    :param logit: A tensor of size BS x H x W
    :return: A tensor of size BS x 2 the soft-argmax in normalized coordinates (-1 .. 1)
    """
    weights = F.softmax(logit.view(logit.size(0), -1), dim=-1).view_as(logit)
    return torch.stack(((weights.sum(1) * torch.linspace(-1, 1, logit.size(2)).to(logit.device)[None]).sum(1),
                        (weights.sum(2) * torch.linspace(-1, 1, logit.size(1)).to(logit.device)[None]).sum(1)), 1)


class HockeyPlayer:
    """
       Your ice hockey player. You may do whatever you want here. There are three rules:
        1. no calls to the pystk library (your code will not run on the tournament system if you do)
        2. There needs to be a deep network somewhere in the loop
        3. You code must run in 100 ms / frame on a standard desktop CPU (no for testing GPU)
        
        Try to minimize library dependencies, nothing that does not install through pip on linux.
    """
    """
       You may request to play with a different kart.
       Call `python3 -c "import pystk; pystk.init(pystk.GraphicsConfig.ld()); print(pystk.list_karts())"` to see all values.
       all_players = ['adiumy', 'amanda', 'beastie', 'emule', 'gavroche', 'gnu', 'hexley', 'kiki', 'konqi', 'nolok', 'pidgin', 
                    'puffy', 'sara_the_racer', 'sara_the_wizard', 'suzanne', 'tux', 'wilber', 'xue']
    """
    kart = ''


    class Model(torch.nn.Module):
        """
        Your code here
        """
        class Block(torch.nn.Module):
            def __init__(self, n_input, n_output, kernel_size=3, stride=2):
                super().__init__()
                self.c1 = torch.nn.Conv2d(n_input, n_output, kernel_size=kernel_size, padding=kernel_size // 2,
                                        stride=stride, bias=False)
                self.c2 = torch.nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
                self.c3 = torch.nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
                self.b1 = torch.nn.BatchNorm2d(n_output)
                self.b2 = torch.nn.BatchNorm2d(n_output)
                self.b3 = torch.nn.BatchNorm2d(n_output)
                self.skip = torch.nn.Conv2d(n_input, n_output, kernel_size=1, stride=stride)

            def forward(self, x):
                return F.relu(self.b3(self.c3(F.relu(self.b2(self.c2(F.relu(self.b1(self.c1(x)))))))) + self.skip(x))

        class UpBlock(torch.nn.Module):
            def __init__(self, n_input, n_output, kernel_size=3, stride=2):
                super().__init__()
                self.c1 = torch.nn.ConvTranspose2d(n_input, n_output, kernel_size=kernel_size, padding=kernel_size // 2,
                                        stride=stride, output_padding=1)

            def forward(self, x):
                return F.relu(self.c1(x))

        def __init__(self, layers=[16, 32, 64, 128], n_output_channels=1, kernel_size=3, use_skip=True):
            super().__init__()
            self.input_mean = torch.Tensor([0.3521554, 0.30068502, 0.28527516])
            self.input_std = torch.Tensor([0.18182722, 0.18656468, 0.15938024])

            c = 3
            self.use_skip = use_skip
            self.n_conv = len(layers)
            skip_layer_size = [3] + layers[:-1]
            for i, l in enumerate(layers):
                self.add_module('conv%d' % i, self.Block(c, l, kernel_size, 2))
                c = l
            for i, l in list(enumerate(layers))[::-1]:
                self.add_module('upconv%d' % i, self.UpBlock(c, l, kernel_size, 2))
                c = l
                if self.use_skip:
                    c += skip_layer_size[i]
            self.classifier = torch.nn.Conv2d(c, n_output_channels, 1)

        def forward(self, x):
            """
            Your code here
            Predict the aim point in image coordinate, given the supertuxkart image
            @img: (B,3,96,128)
            return (B,2)
            """
            z = (x - self.input_mean[None, :, None, None].to(x.device)) / self.input_std[None, :, None, None].to(x.device)
            up_activation = []

            for i in range(self.n_conv):
                # Add all the information required for skip connections
                up_activation.append(z)
                z = self._modules['conv%d'%i](z)

            for i in reversed(range(self.n_conv)):
                z = self._modules['upconv%d'%i](z)
                # Fix the padding
                z = z[:, :, :up_activation[i].size(2), :up_activation[i].size(3)]
                # Add the skip connection
                if self.use_skip:
                    z = torch.cat([z, up_activation[i]], dim=1)
            
            img = self.classifier(z)
            img = torch.squeeze(img, 1)


            return spatial_argmax(img)

    
    def __init__(self, player_id = 0):
        """
        Set up a soccer player.
        The player_id starts at 0 and increases by one for each player added. You can use the player id to figure out your team (player_id % 2), or assign different roles to different agents.
        """
        self.kart = 'xue'
        
    def act(self, image, player_info):
        """
        Set the action given the current image
        :param image: numpy array of shape (300, 400, 3)
        :param player_info: pystk.Player object for the current kart -> https://pystk.readthedocs.io/en/latest/state.html?highlight=pystk.Player#pystk.Player
        return: Dict describing the action
        """
        action = {'acceleration': 0, 'brake': False, 'drift': False, 'nitro': False, 'rescue': False, 'steer': 0}
        """
        Your code here.
        """

        aim_point = Model.forward(image)
        kart_direction = player_info.kart.front
        x = aim_point[0]

        # if current_vel != target_vel:
        #     action.acceleration = 1

        # if x == 0:
        #     action.nitro = True
        
        if x < 0: #aim_point is to the left
            if x < -0.5:
                action['drift'] = True
                action['steer'] = -1
                action['acceleration'] = .25
                action['brake'] = True
            elif x < -0.3:
                action['drift'] = True
                action['steer'] = -1
                action['acceleration'] = .75
            else:
                action['steer'] = -.7

        elif x > 0: #aim_point is to the right
            if x > 0.5:
                action['drift'] = True
                action['steer'] = 1
                action['acceleration'] = .25
                action['brake'] = True
            elif x > 0.3:
                action['drift'] = True
                action['steer'] = 1
                action['acceleration'] = .75
            else:
                action['steer'] = .7


        return action


def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, Planner):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'planner.th'))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model():
    from torch import load
    from os import path
    r = Planner()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'planner.th'), map_location='cpu'))
    return r