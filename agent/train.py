from .model import PuckDetector, save_model
import torch
import torch.utils.tensorboard as tb
import numpy as np
from .utils import load_data
from . import dense_transforms
from torchvision import transforms as T


def train(args):
    from os import path
    import timeit
    from tqdm import tqdm

    # Take the time of training
    start = timeit.default_timer()


    # Do the CUDA
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('device = ', device)
    print(torch.cuda.get_device_name(device))

    # Establish logger and extract env arguements
    rate = args.lrate
    epoch = args.epoch

    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'fcmtrain' + args.run), flush_secs=1)
        params_logger = tb.SummaryWriter(path.join(args.log_dir, 'fcmparams' + args.run), flush_secs=1)

    params_logger.add_text('lrate', str(rate))
    params_logger.add_text('epoch', str(epoch))
    params_logger.add_text('kernel', str(args.kernel))
    # params_logger.add_text('patience', str(args.patience))

    
    # Define model stuff
    # layers = [16, 32, 64, 128]
    # params_logger.add_text('layers', str(layers))
    # model = PuckDetector(layers=layers).to(device)
    model = PuckDetector().to(device)
    
    # Define Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=rate, weight_decay=1e-5)
    loss = torch.nn.MSELoss()
    # loss = torch.nn.BCEWithLogitsLoss()


    # load the data
    print('======================= loading data =======================')
    # we can try this for ColorJitter as well if need be:
    # transformation = dense_transforms.Compose([dense_transforms.ColorJitter(0.4, 0.8, 0.7, 0.3),
    transform = dense_transforms.Compose([dense_transforms.ColorJitter(0.9, 0.9, 0.9, 0.1),
                                           dense_transforms.RandomHorizontalFlip(),
                                           dense_transforms.ToTensor()])                                           

    train_data = load_data(args.trainPath, batch_size=32, transform=transform)

    # run training epochs
    global_step = 0
    print("Setup Complete, starting to train on {epoch} epochs!".format(epoch=epoch))
    for epoch in range(epoch):
        print('======================= training epoch', epoch, '=======================')
        model.train()
        for img, label in tqdm(train_data):
            img, label = img.to(device), label.to(device)
            # print("LABEL PRINT", label.shape)
            img = img.to(device)
            # print("daviceee", img.shape)
            prediction = model(img)
            # print("PREDICTION SIZE", prediction.shape)
            # loss_val = loss(prediction, label)
            # p_sig = torch.sigmoid(prediction * (1-2*label))
            # loss_val = (loss(prediction, label)*p_sig).mean() / p_sig.mean()
            loss_val = loss(prediction, label).to(device)

            # if train_logger is not None and global_step % 100 == 0:
            #     image = torch.cat([img, label], 3).detach().cpu()
            #     train_logger.add_image('image', (torchvision.utils.make_grid(image, padding=5, pad_value=1) * 255))
            
            if train_logger is not None:
                train_logger.add_scalar('loss', loss_val, global_step)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step += 1

    model.eval()
    save_model(model)
    # print("Model shape", type(model))
    train_time = timeit.default_timer()
    print("Done!! Took {}, start {}, {}".format(train_time - start, start, train_time))

        


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', default=r'tmpLogs')
    parser.add_argument('--run', default='6')
    parser.add_argument('-e', '--epoch', default=20)

    # Put custom arguments here
    parser.add_argument('-t', '--trainPath', default=r'data\trainTest')
    parser.add_argument('-l', '--lrate', default=0.01)
    parser.add_argument('--kernel', default=3)

    args = parser.parse_args()
    train(args)
