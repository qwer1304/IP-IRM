import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import STL10, CIFAR10, CIFAR100, ImageFolder
from tqdm import tqdm
import os

import utils
from model import Model

class Net(nn.Module):
    def __init__(self, num_class, pretrained_path, image_class='ImageNet'):
        super(Net, self).__init__()

        # encoder
        model = Model(image_class=image_class).cuda()
        model = Model().cuda()
        model = nn.DataParallel(model)
        model.load_state_dict(torch.load(pretrained_path))

        self.f = model.module.f
        # classifier
        self.fc = nn.Linear(2048, num_class, bias=True)

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.fc(feature)
        return out


# train or test for one epoch
def train_val(net, data_loader, train_optimizer):
    is_train = train_optimizer is not None
    net.train() if is_train else net.eval()

    total_loss, total_correct_1, total_correct_5, total_num, data_bar = 0.0, 0.0, 0.0, 0, tqdm(data_loader)
    with (torch.enable_grad() if is_train else torch.no_grad()):
        for data, target in data_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            out = net(data)
            loss = loss_criterion(out, target)

            if is_train:
                train_optimizer.zero_grad()
                loss.backward()
                train_optimizer.step()

            total_num += data.size(0)
            total_loss += loss.item() * data.size(0)
            prediction = torch.argsort(out, dim=-1, descending=True)
            total_correct_1 += torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_correct_5 += torch.sum((prediction[:, 0:5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

            data_bar.set_description('{} Epoch: [{}/{}] Loss: {:.4f} ACC@1: {:.2f}% ACC@5: {:.2f}%'
                                     .format('Train' if is_train else 'Test', epoch, epochs, total_loss / total_num,
                                             total_correct_1 / total_num * 100, total_correct_5 / total_num * 100))

    return total_loss / total_num, total_correct_1 / total_num * 100, total_correct_5 / total_num * 100


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Linear Evaluation')
    parser.add_argument('--model_path', type=str, default='results/model_400.pth',
                        help='The pretrained model path')
    parser.add_argument('--batch_size', type=int, default=512, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', type=int, default=100, help='Number of sweeps over the dataset to train')
    parser.add_argument('--dataset', type=str, default='STL', choices=['STL', 'CIFAR10', 'CIFAR100', 'ImageNet'], help='experiment dataset')
    parser.add_argument('--data', metavar='DIR', help='path to dataset')
    parser.add_argument('--txt', action="store_true", default=False, help='save txt?')
    parser.add_argument('--name', type=str, default='None', help='exp name?')

    # image
    parser.add_argument('--image_size', type=int, default=224, help='image size')
    # color in label
    parser.add_argument('--target_transform', type=str, default=None, help='a function definition to apply to target')
    parser.add_argument('--image_class', choices=['ImageNet', 'STL', 'CIFAR'], default='ImageNet', help='Image class, default=ImageNet')
    parser.add_argument('--class_num', default=1000, type=int, help='num of classes')
    args = parser.parse_args()

    if not os.path.exists('downstream/{}/{}'.format(args.dataset, args.name)):
        os.makedirs('downstream/{}/{}'.format(args.dataset, args.name))

    # seed
    utils.set_seed(1234)

    model_path, batch_size, epochs = args.model_path, args.batch_size, args.epochs
    target_transform = eval(args.target_transform) if args.target_transform is not None else None
    image_class, image_size = args.image_class, args.image_size

    if args.dataset == 'STL':
        train_transform = utils.make_train_transform()
        train_data = STL10(root=args.data, split='train', transform=train_transform, target_transform=target_transform)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        test_transform = utils.make_test_transform()
        test_data = STL10(root=args.data, split='test', transform=test_transform, target_transform=target_transform)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    elif args.dataset == 'CIFAR10':
        train_transform = utils.make_train_transform()
        train_data = utils.CIFAR10(root=args.data, train=True, transform=train_transform, target_transform=target_transform)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        test_transform = utils.make_test_transform()
        test_data = utils.CIFAR10(root=args.data, train=False, transform=train_transform, target_transform=target_transform)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    elif args.dataset == 'CIFAR100':
        train_transform = utils.make_train_transform()
        train_data = utils.CIFAR100(root=args.data, train=True, transform=train_transform, target_transform=target_transform)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        test_transform = utils.make_test_transform()
        test_data = utils.CIFAR100(root=args.data, train=False, transform=train_transform, target_transform=target_transform)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    elif args.dataset == 'ImageNet':
        train_transform = utils.make_train_transform(image_size)
        train_data = utils.Imagenet_idx_pair(root=args.data+'/train', transform=train_transform, target_transform=target_transform)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        test_transform = utils.make_test_transform()
        test_data = utils.Imagenet_pair(root=args.data+'/testgt', transform=test_transform, target_transform=target_transform)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    num_class = len(train_data.classes) if args.dataset != "ImageNet" else args.class_num
    model = Net(num_class=num_class, pretrained_path=model_path, image_class=image_class).cuda()
    for param in model.f.parameters():
        param.requires_grad = False
    model = nn.DataParallel(model)

    optimizer = optim.Adam(model.module.fc.parameters(), lr=1e-3, weight_decay=1e-6)
    loss_criterion = nn.CrossEntropyLoss()
    results = {'train_loss': [], 'train_acc@1': [], 'train_acc@5': [],
               'test_loss': [], 'test_acc@1': [], 'test_acc@5': []}

    for epoch in range(1, epochs + 1):
        train_loss, train_acc_1, train_acc_5 = train_val(model, train_loader, optimizer)
        test_loss, test_acc_1, test_acc_5 = train_val(model, test_loader, None)

        if args.txt:
            txt_write = open("downstream/{}/{}/{}".format(args.dataset, args.name, 'result.txt'), 'a')
            txt_write.write('\ntest_loss: {}, test_acc@1: {}, test_acc@5: {}'.format(test_loss, test_acc_1, test_acc_5))