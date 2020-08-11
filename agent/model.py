import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms as T


# ============================================== PuckDetector ============================================== 
def spatial_argmax(logit):
    """
    Compute the soft-argmax of a heatmap
    :param logit: A tensor of size BS x H x W
    :return: A tensor of size BS x 2 the soft-argmax in normalized coordinates (-1 .. 1)
    """
    weights = F.softmax(logit.view(logit.size(0), -1), dim=-1).view_as(logit)
    return torch.stack(((weights.sum(1) * torch.linspace(-1, 1, logit.size(2)).to(logit.device)[None]).sum(1),
                        (weights.sum(2) * torch.linspace(-1, 1, logit.size(1)).to(logit.device)[None]).sum(1)), 1)


class PuckDetector(torch.nn.Module):
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
            Predict the aim point in image coordinate, given the supertuxkart image
            @img: (B,3,300,400)
            return (B,2)
            """
            # print("X SHAPE = ", x.shape)
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

            return spatial_argmax(self.classifier(z).squeeze(1))


# # ============================================== OpponentDetector ==============================================

# def extract_peak(heatmap, max_pool_ks=7, min_score=-5, max_det=100):
#     """
#        Extract local maxima (peaks) in a 2d heatmap.
#        @heatmap: H x W heatmap containing peaks (similar to your training heatmap)
#        @max_pool_ks: Only return points that are larger than a max_pool_ks x max_pool_ks window around the point
#        @min_score: Only return peaks greater than min_score
#        @return: List of peaks [(score, cx, cy), ...], where cx, cy are the position of a peak and score is the
#                 heatmap value at the peak. Return no more than max_det peaks per image
#     """
#     max_cls = F.max_pool2d(heatmap[None, None], kernel_size=max_pool_ks, padding=max_pool_ks // 2, stride=1)[0, 0]
#     possible_det = heatmap - (max_cls > heatmap).float() * 1e5
#     if max_det > possible_det.numel():
#         max_det = possible_det.numel()
#     score, loc = torch.topk(possible_det.view(-1), max_det)
#     return [(float(s), int(l) % heatmap.size(1), int(l) // heatmap.size(1))
#             for s, l in zip(score.cpu(), loc.cpu()) if s > min_score]


# class OpponentDetector(torch.nn.Module):
#         class Block(torch.nn.Module):
#             def __init__(self, n_input, n_output, kernel_size=3, stride=2):
#                 super().__init__()
#                 self.c1 = torch.nn.Conv2d(n_input, n_output, kernel_size=kernel_size, padding=kernel_size // 2,
#                                         stride=stride, bias=False)
#                 self.c2 = torch.nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
#                 self.c3 = torch.nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
#                 self.b1 = torch.nn.BatchNorm2d(n_output)
#                 self.b2 = torch.nn.BatchNorm2d(n_output)
#                 self.b3 = torch.nn.BatchNorm2d(n_output)
#                 self.skip = torch.nn.Conv2d(n_input, n_output, kernel_size=1, stride=stride)

#             def forward(self, x):
#                 return F.relu(self.b3(self.c3(F.relu(self.b2(self.c2(F.relu(self.b1(self.c1(x)))))))) + self.skip(x))

#         class UpBlock(torch.nn.Module):
#             def __init__(self, n_input, n_output, kernel_size=3, stride=2):
#                 super().__init__()
#                 self.c1 = torch.nn.ConvTranspose2d(n_input, n_output, kernel_size=kernel_size, padding=kernel_size // 2,
#                                         stride=stride, output_padding=1)

#             def forward(self, x):
#                 return F.relu(self.c1(x))

#         def __init__(self, layers=[16, 32, 64, 128], n_output_channels=3, kernel_size=3, use_skip=True):
#             super().__init__()
#             self.input_mean = torch.Tensor([0.3521554, 0.30068502, 0.28527516])
#             self.input_std = torch.Tensor([0.18182722, 0.18656468, 0.15938024])

#             c = 3
#             self.use_skip = use_skip
#             self.n_conv = len(layers)
#             skip_layer_size = [3] + layers[:-1]
#             for i, l in enumerate(layers):
#                 self.add_module('conv%d' % i, self.Block(c, l, kernel_size, 2))
#                 c = l
#             for i, l in list(enumerate(layers))[::-1]:
#                 self.add_module('upconv%d' % i, self.UpBlock(c, l, kernel_size, 2))
#                 c = l
#                 if self.use_skip:
#                     c += skip_layer_size[i]
#             self.classifier = torch.nn.Conv2d(c, n_output_channels, 1)

#         def forward(self, x):
#             """
#             Your code here.
#             Implement object detection here.
#             @image: 3 x H x W image
#             @return: Three list of detections [(score, cx, cy, w/2, h/2), ...], one per class,
#                     return no more than 30 detections per image per class. You only need to predict width and height
#                     for extra credit. If you do not predict an object size, return w=0, h=0.
#             Hint: Use extract_peak here
#             Hint: Make sure to return three python lists of tuples of (float, int, int, float, float) and not a pytorch
#                     scalar. Otherwise pytorch might keep a computation graph in the background and your program will run
#                     out of memory.
#             """
#             z = (x - self.input_mean[None, :, None, None].to(x.device)) / self.input_std[None, :, None, None].to(x.device)
#             up_activation = []

#             for i in range(self.n_conv):
#                 # Add all the information required for skip connections
#                 up_activation.append(z)
#                 z = self._modules['conv%d'%i](z)

#             for i in reversed(range(self.n_conv)):
#                 z = self._modules['upconv%d'%i](z)
#                 # Fix the padding
#                 z = z[:, :, :up_activation[i].size(2), :up_activation[i].size(3)]
#                 # Add the skip connection
#                 if self.use_skip:
#                     z = torch.cat([z, up_activation[i]], dim=1)

#             cls = self.classifier(z)

#             return [[(s, x, y) for s, x, y in extract_peak(c, max_det=4)] for c in cls[0]]


def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, PuckDetector):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'model.th'))
    raise ValueError("model type '%s' not supported!" % str(type(model)))

def load_model():
    from torch import load
    from os import path
    r = PuckDetector()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'model.th'), map_location='cpu'))
    return r
