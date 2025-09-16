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

from prepare import prepare_datasets, traverse_objects

class Net(nn.Module):
    def __init__(self, num_class, pretrained_path, image_class='ImageNet', args=None):
        super(Net, self).__init__()

        # encoder
        model = Model(image_class=image_class).cuda()
        model = Model().cuda()
        model = nn.DataParallel(model)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        msg = []
        assert (pretrained_path is not None and os.path.isfile(pretrained_path))
        print("=> loading pretrained checkpoint '{}'".format(pretrained_path))
        checkpoint = torch.load(pretrained_path, map_location=device, weights_only=False)
        if 'state_dict' in checkpoint.keys():
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        self.f = model.module.f
        if args.evaluate is None or args.evaluate == 'knn':
            msg = model.load_state_dict(state_dict, strict=False)
            print(msg)
            # classifier
            self.fc = nn.Linear(2048, num_class, bias=True)
        else:
            model.module.fc = nn.Linear(2048, num_class, bias=True)
            msg = model.load_state_dict(state_dict, strict=False)
            missing_keys = [k for k in msg.missing_keys if 'g.' not in k]
            if msg.unexpected_keys or missing_keys:
                print(msg.unexpected_keys, missing_keys)
            self.fc = model.module.fc

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.fc(feature)
        return out, feature


# train or test for one epoch
def train_val(net, data_loader, train_optimizer, batch_size, args, dataset="test"):
    is_train = train_optimizer is not None
    net.train() if is_train else net.eval()
    
    transform = data_loader.dataset.transform
    target_transform = data_loader.dataset.target_transform
    
    gradients_batch_size = args.gradients_batch_size
    loader_batch_size = batch_size
    gpu_batch_size = args.micro_batch_size
    
    loader_accum_steps = gradients_batch_size // loader_batch_size 
    gpu_accum_steps = loader_batch_size // gpu_batch_size 
    
    loader_step = 0
    total_samples = len(data_loader.dataset)
    
    total_loss, total_correct_1, total_correct_5, total_num = 0.0, 0.0, 0.0, 0
    bar_format = '{l_bar}{bar:' + str(args.bar) + '}{r_bar}' #{bar:-' + str(args.bar) + 'b}'
    data_bar = tqdm(data_loader,
            total=len(data_loader),
            ncols=args.ncols,               # total width available
            dynamic_ncols=False,            # disable autosizing
            bar_format=bar_format,          # request bar width
            )
    with (torch.enable_grad() if is_train else torch.no_grad()):
        if args.extract_features:
            data_loader.dataset.target_transform = None

        feature_list = []
        pred_labels_list = []
        pred_scores_list = []
        target_list = []
        target_raw_list = []

        if is_train:
            train_optimizer.zero_grad()  # clear gradients at the beginning

        for batch_data in data_bar:
            data, target = batch_data[0], batch_data[1] # ommit index, if returned 
            # Split into micro-batches
            if gpu_accum_steps > 1:
                data_chunks = data.chunk(gpu_accum_steps)
                target_chunks = target.chunk(gpu_accum_steps)
            else:
                data_chunks = (data)
                target_chunks = (target)
            
            for data_chunk, target_chunk in zip(data_chunks, target_chunks):
                data, target = data_chunk.cuda(non_blocking=True), target_chunk.cuda(non_blocking=True)

                target_raw = target
                if args.extract_features and target_transform is not None:
                    target = target_transform(target_raw).cuda(non_blocking=True)

                if transform is not None:
                    data = transform(data)

                out, feature = net(data)
                loss = loss_criterion(out, target)

                loss = loss / gpu_accum_steps / loader_accum_steps  # scale loss to account for accumulation

                if is_train:
                    loss.backward() # adds gradients to accumulated ones
            
                total_num += data.size(0)
                total_loss += loss.item() * data.size(0)
                prediction = torch.argsort(out, dim=-1, descending=True)
                total_correct_1 += torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
                total_correct_5 += torch.sum((prediction[:, 0:5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

                # compute output
                if args.extract_features:
                    feature_list.append(feature)
                    target_list.append(target)
                    target_raw_list.append(target_raw)
                    pred_labels_list.append(prediction)
                    pred_scores_list.append(out)

                # free memory of micro-batch
                del data, target, loss
                torch.cuda.empty_cache()

            # end for data_chunk, target_chunk in zip(data_chunks, target_chunks):
    
            loader_step += 1
            if (loader_step * loader_batch_size) == gradients_batch_size:
                loader_step = 0
                if is_train:
                    train_optimizer.step()
                    train_optimizer.zero_grad()  # clear gradients at beginning of next gradients batch

            data_bar.set_description('{} Epoch: [{}/{}] Loss: {:.4f} Acc@1: {:.2f}% Acc@5: {:.2f}%'
                                     .format(dataset.capitalize(), epoch, epochs, total_loss / total_num,
                                             total_correct_1 / total_num * 100, total_correct_5 / total_num * 100))
        # end for data, target in data_bar:

        if feature_list:
            feature = torch.cat(feature_list, dim=0)
            target = torch.cat(target_list, dim=0)
            target_raw = torch.cat(target_raw_list, dim=0)
            pred_labels = torch.cat(pred_labels_list, dim=0)
            pred_scores = torch.cat(pred_scores_list, dim=0)

            # Save to file
            prefix = "test"
            directory = f'downstream/{args.dataset}/{args.name}'
            fp = os.path.join(directory, f"{dataset}_features_dump.pt")       
            os.makedirs(os.path.dirname(fp), exist_ok=True)

            torch.save({
                'features':     feature,
                'labels':       target,
                'labels_raw':   target_raw,
                'pred_labels':  pred_labels,
                'pred_scores':  pred_scores,
                'model_epoch':  epoch,
                'n_classes':    args.class_num,
            }, fp)
            print(f"Dumped features into {fp}")
        
    return total_loss / total_num, total_correct_1 / total_num * 100, total_correct_5 / total_num * 100


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Linear Evaluation')
    parser.add_argument('--model_path', type=str, default='results/model_400.pth',
                        help='The pretrained model path')
    parser.add_argument('--dl_tr', default=[256, 4, 2, True], nargs=4, type=str,
                        action=utils.ParseMixed, types=[int, int, int, bool],
                        metavar='DataLoader pars [batch_size, number_workers, prefetch_factor, persistent_workers]', help='Training minimization DataLoader pars')
    parser.add_argument('--dl_te', default=[3096, 4, 2, 1], nargs=4, type=str,
                        action=utils.ParseMixed, types=[int, int, int, bool],
                        metavar='DataLoader pars [batch_size, number_workers, prefetch_factor, persistent_workers]', help='Testing/Validation/Memory DataLoader pars')
    parser.add_argument('--micro_batch_size', default=32, type=int, help='batch size on gpu')
    parser.add_argument('--gradients_batch_size', default=256, type=int, help='batch size of gradients accumulation')
    parser.add_argument('--epochs', type=int, default=100, help='Number of sweeps over the dataset to train')
    parser.add_argument('--dataset', type=str, default='STL', choices=['STL', 'CIFAR10', 'CIFAR100', 'ImageNet'], help='experiment dataset')
    parser.add_argument('--data', metavar='DIR', help='path to dataset')
    parser.add_argument('--txt', action="store_true", default=False, help='save txt?')
    parser.add_argument('--name', type=str, default='None', help='exp name?')

    # image
    parser.add_argument('--image_size', type=int, default=224, help='image size')
    # color in label
    parser.add_argument('--target_transform', type=str, default=None, help='a function definition to apply to target')
    parser.add_argument('--class_to_idx', type=str, default=None, help='a function definition to apply to class to obtain it index')
    parser.add_argument('--image_class', choices=['ImageNet', 'STL', 'CIFAR'], default='ImageNet', help='Image class, default=ImageNet')
    parser.add_argument('--class_num', default=1000, type=int, help='num of classes')
    parser.add_argument('--train_envs', type=str, nargs='+', default=None, required=True)
    parser.add_argument('--test_envs', type=str, nargs='+', default=None)
    parser.add_argument('--holdout_fraction', type=float, default=0.8)

    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--ncols', default=80, type=int, help='number of columns in terminal')
    parser.add_argument('--bar', default=50, type=int, help='length of progess bar')

    parser.add_argument('--norandgray', action="store_true", default=False, help='skip rand gray transform')
    parser.add_argument('--evaluate', type=str, default=None, choices=['knn', 'linear'], help='only evaluate')
    parser.add_argument('--extract_features', action="store_true", help="extract features for post processiin during evaluate")

    args = parser.parse_args()

    if not os.path.exists('downstream/{}/{}'.format(args.dataset, args.name)):
        os.makedirs('downstream/{}/{}'.format(args.dataset, args.name))

    # seed
    utils.set_seed(args.seed)

    model_path, epochs = args.model_path, args.epochs
    dl_tr, dl_te = args.dl_tr, args.dl_te
    target_transform = eval(args.target_transform) if args.target_transform is not None else None
    class_to_idx = eval(args.class_to_idx) if args.class_to_idx is not None else None
    image_class, image_size = args.image_class, args.image_size

    # data prepare
    tr_bs, tr_nw, tr_pf, tr_pw = dl_tr
    te_bs, te_nw, te_pf, te_pw = dl_te
    if args.dataset == 'STL':
        train_transform = utils.make_train_transform(normalize=args.image_class)
        test_transform = utils.make_test_transform(normalize=args.image_class)
        train_data = STL10(root=args.data, split='train', transform=train_transform, target_transform=target_transform)
        test_data = STL10(root=args.data, split='test', transform=test_transform, target_transform=target_transform)
        train_loader = DataLoader(train_data, batch_size=tr_bs, num_workers=tr_nw, prefetch_factor=tr_pf, shuffle=True, pin_memory=True, 
            drop_last=True, persistent_workers=tr_pw)
        test_loader = DataLoader(test_data, batch_size=te_bs, num_workers=te_nw, prefetch_factor=te_pf, shuffle=False, 
            pin_memory=True, persistent_workers=te_pw)
    elif args.dataset == 'CIFAR10':
        train_transform = utils.make_train_transform(normalize=args.image_class)
        test_transform = utils.make_test_transform(normalize=args.image_class)
        train_data = utils.CIFAR10(root=args.data, train=True, transform=train_transform, target_transform=target_transform)
        test_data = utils.CIFAR10(root=args.data, train=False, transform=train_transform, target_transform=target_transform)
        train_loader = DataLoader(train_data, batch_size=tr_bs, num_workers=tr_nw, prefetch_factor=tr_pf, shuffle=True, pin_memory=True, 
            drop_last=True, persistent_workers=tr_pw)
        test_loader = DataLoader(test_data, batch_size=te_bs, num_workers=te_nw, prefetch_factor=te_pf, shuffle=False, 
            pin_memory=True, persistent_workers=te_pw)
    elif args.dataset == 'CIFAR100':
        train_transform = utils.make_train_transform(normalize=args.image_class)
        test_transform = utils.make_test_transform(normalize=args.image_class)
        train_data = utils.CIFAR100Pair_Index(root=args.data, train=True, transform=train_transform, target_transform=target_transform)
        test_data = utils.CIFAR100Pair(root=args.data, train=False, transform=test_transform, target_transform=target_transform)
        train_loader = DataLoader(train_data, batch_size=tr_bs, num_workers=tr_nw, prefetch_factor=tr_pf, shuffle=True, pin_memory=True, 
            drop_last=True, persistent_workers=tr_pw)
        test_loader = DataLoader(test_data, batch_size=te_bs, num_workers=te_nw, prefetch_factor=te_pf, shuffle=False, 
            pin_memory=True, persistent_workers=te_pw)
    elif args.dataset == 'ImageNet':
        train_transform = utils.make_train_transform(image_size, randgray=not args.norandgray, normalize=args.image_class)
        test_transform = utils.make_test_transform(normalize=args.image_class)

        if False:
            wrap = args.extract_features
            # descriptors of train data
            train_desc  =   {'dataset': utils.Imagenet_idx,
                              'transform': train_transform,
                              'target_transform': target_transform,
                              'class_to_index': class_to_idx,
                              'wrap': False, # for changeable target transform
                              'required_split': "in",
                            }
            # descriptors of test data
            test_desc   =   {'dataset': utils.Imagenet,
                              'transform': test_transform,
                              'target_transform': target_transform,
                              'class_to_index': class_to_idx,
                              'wrap': wrap, # for changeable target transform
                              'required_split': "in",
                            }

            datas = prepare_datasets(args.data, args.train_envs, [train_desc], args.holdout_fraction, args.seed)
            train_data = datas[0][0]

            datas = prepare_datasets(args.data, args.test_envs, [test_desc], 1.0, args.seed)
            test_data = datas[0][0]

            #traverse_objects(update_data)
            #exit()

        else:
            train_data  = utils.Imagenet_idx(root=args.data + '/train', transform=train_transform, target_transform=target_transform, class_to_idx=class_to_idx)
            test_data   = utils.Imagenet(root=args.data     + '/test',  transform=test_transform,  target_transform=target_transform, class_to_idx=class_to_idx)
            val_data    = utils.Imagenet(root=args.data     + '/val',   transform=test_transform,  target_transform=target_transform, class_to_idx=class_to_idx)

        train_loader = DataLoader(train_data, batch_size=tr_bs, num_workers=tr_nw, prefetch_factor=tr_pf, shuffle=True, pin_memory=True, 
            drop_last=True, persistent_workers=tr_pw)
        test_loader = DataLoader(test_data, batch_size=te_bs, num_workers=te_nw, prefetch_factor=te_pf, shuffle=False, 
            pin_memory=True, persistent_workers=te_pw)
        val_loader = DataLoader(val_data, batch_size=te_bs, num_workers=te_nw, prefetch_factor=te_pf, shuffle=False, 
            pin_memory=True, persistent_workers=te_pw)

    num_class = len(train_data.classes) if args.dataset != "ImageNet" else args.class_num
    
    class FeatureQueue: # to make checkpoint load happy
        def __init__(self, *args, **kwargs): pass
        
    model = Net(num_class=num_class, pretrained_path=model_path, image_class=image_class, args=args).cuda()
    for param in model.f.parameters():
        param.requires_grad = False
    model = nn.DataParallel(model)

    optimizer = optim.Adam(model.module.fc.parameters(), lr=1e-3, weight_decay=1e-6)
    loss_criterion = nn.CrossEntropyLoss()
    results = {'train_loss': [], 'train_acc@1': [], 'train_acc@5': [],
               'test_loss': [], 'test_acc@1': [], 'test_acc@5': []}
    if args.dataset == 'ImageNet':
        results.update({'val_loss': [], 'val_acc@1': [], 'val_acc@5': []})

    if args.evaluate:
        epoch = epochs
        train_loss, train_acc_1, train_acc_5 = train_val(model, train_loader, None, tr_bs, args, dataset="train")
        test_loss, test_acc_1, test_acc_5 = train_val(model, test_loader, None, te_bs, args, dataset="test")
        if args.dataset == 'ImageNet':
            val_loss, val_acc_1, val_acc_5 = train_val(model, val_loader, None, te_bs, args, dataset="val")
        if args.txt:
            txt_write = open("downstream/{}/{}/{}".format(args.dataset, args.name, 'result.txt'), 'a')
            txt_write.write('\ntest_loss: {}, test_acc@1: {}, test_acc@5: {}'.format(test_loss, test_acc_1, test_acc_5))
            if args.dataset == 'ImageNet':
                txt_write.write('\nval_loss: {}, val_acc@1: {}, val_acc@5: {}'.format(val_loss, val_acc_1, val_acc_5))
    
    else:
        for epoch in range(1, epochs + 1):
            train_loss, train_acc_1, train_acc_5 = train_val(model, train_loader, optimizer, tr_bs, args, dataset="train")
            if args.dataset == 'ImageNet':
                val_loss, val_acc_1, val_acc_5 = train_val(model, val_loader, None, te_bs, args, dataset="val")
            test_loss, test_acc_1, test_acc_5 = train_val(model, test_loader, None, te_bs, args, dataset="test")

            if args.txt:
                txt_write = open("downstream/{}/{}/{}".format(args.dataset, args.name, 'result.txt'), 'a')
                txt_write.write('\ntest_loss: {}, test_acc@1: {}, test_acc@5: {}'.format(test_loss, test_acc_1, test_acc_5))
                if args.dataset == 'ImageNet':
                    txt_write.write('\nval_loss: {}, val_acc@1: {}, val_acc@5: {}'.format(val_loss, val_acc_1, val_acc_5))

        torch.save(model.state_dict(), 'downstream/{}/{}/model_{}.pth'.format(args.dataset, args.name, epoch))
