from .model import Model, save_model
import torch
import torch.utils.tensorboard as tb
import numpy as np
from .utils import load_data
from . import dense_transforms
from torchvision import transforms as T


def train(args):
    from os import path
    model = Model()
    train_logger = None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    loss = torch.nn.MSELoss(reduction='none')

    # load the data
    print('======================= loading data =======================')
    transform = dense_transforms.Compose([dense_transforms.ColorJitter(0.9, 0.9, 0.9, 0.1),
                                           dense_transforms.RandomHorizontalFlip(),
                                           dense_transforms.ToTensor()])                                           

    train_data = load_data(num_workers=4, transform=transform)


    # run training epochs
    global_step = 0
    for epoch in range(args.num_epoch):
        print('======================= training epoch', epoch, '=======================')
        
        model.train()
        for img, aim_pt in train_data:
            img, aim_pt = img.to(device), aim_pt.to(device)

            prediction = model(img)
            # loss_val = loss(prediction, aim_pt)
            p_sig = torch.sigmoid(prediction * (1-2*aim_pt))
            loss_val = (loss(prediction, aim_pt)*p_sig).mean() / p_sig.mean()

            if train_logger is not None and global_step % 100 == 0:
                image = torch.cat([img, aim_pt], 3).detach().cpu()
                train_logger.add_image('image', (torchvision.utils.make_grid(image, padding=5, pad_value=1) * 255))
            
            if train_logger is not None:
                train_logger.add_scalar('loss', loss_val, global_step)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step += 1

        save_model(model)

        


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('-n', '--num_epoch', type=int, default=60)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    # parser.add_argument('-c', '--continue_training', action='store_true')

    args = parser.parse_args()
    train(args)