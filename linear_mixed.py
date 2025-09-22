import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import STL10, CIFAR10, CIFAR100, ImageFolder
import random
import shutil
import numpy as np
from tqdm import tqdm
import os
from collections import defaultdict, OrderedDict

import utils
from model import Model

from prepare import prepare_datasets, traverse_objects

import kornia.augmentation as K


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
        
        def rename_key_from_standard(k, keepfc=False):
            # Skip fc, since your model doesn't use it
            if (not keepfc) and k.startswith("module.fc."):
                return None

            new_k = k
            new_k = new_k.replace("module.conv1.", "module.f.0.")
            new_k = new_k.replace("module.bn1.", "module.f.1.")
            new_k = new_k.replace("module.layer1.", "module.f.4.")
            new_k = new_k.replace("module.layer2.", "module.f.5.")
            new_k = new_k.replace("module.layer3.", "module.f.6.")
            new_k = new_k.replace("module.layer4.", "module.f.7.")
            return new_k

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            # Remove "module.model." prefix
            name = k.replace("module.encoder_q.", "module.")
            #name = name.replace("module.", "")                  
            new_state_dict[name] = v
        state_dict = new_state_dict
        
        # convert pretrained dict
        converted_dict = {}
        for k, v in state_dict.items():
            new_k = rename_key_from_standard(k, keepfc=(args.evaluate == 'linear'))
            if new_k is not None:  # skip fc
                converted_dict[new_k] = v

        state_dict = converted_dict

        msg = model.load_state_dict(state_dict, strict=False)
        missing_keys = [k for k in msg.missing_keys if 'g.' not in k]
        if msg.unexpected_keys or missing_keys:
            print(msg.unexpected_keys)
            print(missing_keys)
        if args.evaluate is None or args.evaluate == 'knn':
            # If training or evaluating output from SSL
            # classifier
            self.fc = nn.Linear(2048, num_class, bias=True)
        else:
            self.fc = model.module.fc

    def forward(self, x, normalize=False):
        with torch.no_grad():
            x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        if normalize:
            feature = F.normalize(feature, dim=1)  # L2 norm
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

    """
    mixup = K.RandomMixUpV2(
        lambda_val=torch.tensor([0.3, 0.7]),  # Beta distribution parameter, [min,max]
        same_on_batch=False,                  # different lambda per sample
        p=1.0,                                # apply to all samples
        keepdim=False,                        # output same shape as input
        data_keys=["input", "target"]         # specify which tensors to mix
    )
    """
    
    mixup = K.RandomMixUpV2(data_keys=["input", "class"], same_on_batch=False, keepdim=True,)       
    cutmix = K.RandomCutMixV2(data_keys=["input", "class"], same_on_batch=False, keepdim=True,)
    mix_list = [mixup, cutmix]    
    
    loss_mixup_criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing, reduction='none')

    with (torch.enable_grad() if is_train else torch.no_grad()):
        if args.extract_features:
            data_loader.dataset.target_transform = None

        feature_mix_list = []
        target_mix_list = []

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
                data_chunks = data.split(gpu_batch_size)
                target_chunks = target.split(gpu_batch_size)
            else:
                data_chunks = (data,)
                target_chunks = (target,)
            
            for data_chunk, target_chunk in zip(data_chunks, target_chunks):
                data, target = data_chunk.cuda(non_blocking=True), target_chunk.cuda(non_blocking=True)

                target_raw = target
                if args.extract_features and target_transform is not None:
                    target = target_transform(target_raw).cuda(non_blocking=True)

                if transform is not None:
                    data = transform(data)

                out, feature = net(data, normalize=True)
                loss = loss_criterion(out, target)

                total_num += data.size(0)
                total_loss += loss.item() * data.size(0)
                prediction = torch.argsort(out, dim=-1, descending=True)
                total_correct_1 += torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
                total_correct_5 += torch.sum((prediction[:, 0:5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

                feature_mix_list.append(feature)
                target_mix_list.append(target)

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
                    feature = torch.cat(feature_mix_list, dim=0)
                    target = torch.cat(target_mix_list, dim=0)
                    feature_mix_list, target_mix_list = [], []
                    feat_chunks = torch.chunk(feature, 2)
                    target_chunks = torch.chunk(target, 2)
                    for i in range(2):
                        feature = feat_chunks[i].unsqueeze(1).unsqueeze(2)
                        target = target_chunks[i]
                        feature_mixed, labels_mixed = mix_list[i](feature, target)
                        feature_mixed, labels_mixed = feature_mixed.squeeze(), labels_mixed.squeeze()
                        feature_mix_list.append(feature_mixed)
                        target_mix_list.append(labels_mixed)
                    feature_mixed = torch.cat(feature_mix_list, dim=0)
                    labels_mixed = tuple(torch.cat(labels, dim=0) for labels in zip(*target_mix_list))
                    out = net.module.fc(feature_mixed)
                    def loss_mixup(y, logits):
                        loss_a = loss_mixup_criterion(logits, y[:, 0].long())
                        loss_b = loss_mixup_criterion(logits, y[:, 1].long())
                        return ((1 - y[:, 2]) * loss_a + y[:, 2] * loss_b).mean()
    
                    loss = loss_mixup(labels_mixed, out)
                    loss.backward()
            
                    train_optimizer.step()
                    train_optimizer.zero_grad()  # clear gradients at beginning of next gradients batch
                    feature_mix_list, target_mix_list = [], []

            data_bar.set_description('{} Epoch: [{}/{}] [{}/{}] Loss: {:.4f} Acc@1: {:.2f}% Acc@5: {:.2f}%'
                                     .format(dataset.capitalize(), epoch, epochs, total_num, len(data_loader.dataset),
                                             total_loss / total_num,
                                             total_correct_1 / total_num * 100, total_correct_5 / total_num * 100))
        # end for data, target in data_bar:

        if args.extract_features:
            feature = torch.cat(feature_list, dim=0)
            target = torch.cat(target_list, dim=0)
            target_raw = torch.cat(target_raw_list, dim=0)
            pred_labels = torch.cat(pred_labels_list, dim=0)
            pred_scores = torch.cat(pred_scores_list, dim=0)

            # Save to file
            prefix = "test"
            directory = f'downstream/{args.name}'
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

def save_checkpoint(state, is_best, args, filename='checkpoint.pth.tar', sync=True):
    filename_tmp = filename + ".tmp"
    torch.save(state, filename_tmp)
    os.replace(filename_tmp, filename)
    if is_best:
        best_filename = '{}/model_best.pth.tar'.format(os.path.dirname(filename))
        best_filename_tmp = filename + ".tmp"
        shutil.copyfile(filename, best_filename_tmp)
        os.replace(best_filename_tmp, best_filename)
    if sync:
        # Sync file data
        for p in [filename, best_filename] if is_best else [filename]:
            fd = os.open(p, os.O_RDONLY)
            os.fsync(fd)
            os.close(fd)

        # Sync the directory once (covers both files)
        dir_fd = os.open(os.path.dirname(filename) or ".", os.O_RDONLY)
        os.fsync(dir_fd)
        os.close(dir_fd)

def load_checkpoint(path, model, optimizer, device='cuda'):
    print("=> loading checkpoint '{}'".format(path))
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    # Restore training bookkeeping
    start_epoch = checkpoint['epoch'] + 1
    best_acc1 = checkpoint['best_acc1']
    best_epoch = checkpoint['best_epoch']

    # Restore models
    msg_model = model.load_state_dict(checkpoint['state_dict'])
    # Restore optimizer
    optimizer.load_state_dict(checkpoint['optimizer']) # nothing ia returned
    # Ensure optimizer state tensors are on the right device
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)
                
    # Restore RNG states
    rng_dict = checkpoint['rng_dict']
    rng_state = rng_dict['rng_state']
    if rng_state.device != torch.device('cpu'):
        rng_state = rng_state.cpu()   
    torch.set_rng_state(rng_state)
    if rng_dict['cuda_rng_state'] is not None:
        torch.cuda.set_rng_state_all([t.cpu() if t.device != torch.device('cpu') else t for t in rng_dict['cuda_rng_state']])
    np.random.set_state(rng_dict['numpy_rng_state'])
    random.setstate(rng_dict['python_rng_state'])

    print(f"\tmodel load: {msg_model}")
    print("<= loaded checkpoint '{}' (epoch {})"
          .format(path, checkpoint['epoch']))

    return model, optimizer, start_epoch, best_acc1, best_epoch

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

    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
    parser.add_argument('--checkpoint_freq', default=3, type=int, metavar='N',
                    help='checkpoint epoch freqeuncy')   
    parser.add_argument('--val_freq', default=1, type=int, metavar='N',
                    help='validation epoch freqeuncy')   
    parser.add_argument('--test_freq', default=None, type=int, metavar='N',
                    help='test epoch freqeuncy')   
    parser.add_argument('--lr', default=0.001, type=float, help='LR')
    parser.add_argument('--weight_decay', default=1e-6, type=float, help='weight decay')
    parser.add_argument('--label_smoothing', default=0.0, type=float, help='label smoothing')
    parser.add_argument('--prune_sizes', action="store_true", help="prune training dataset to minority class size")
    parser.add_argument('--weighted_loss', action="store_true", help="weight each sample by its class size")

    args = parser.parse_args()

    save_dir = 'downstream/{}'.format(args.name)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

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
            train_data  = utils.Imagenet(root=args.data + '/train', transform=train_transform, target_transform=target_transform, class_to_idx=class_to_idx)
            if args.prune_sizes:
                class SubsetProxy(Subset):
                    def __getattr__(self, name):
                        # called only if attribute not found in self
                        return getattr(self.dataset, name)

                    def __setattr__(self, name, value):
                        if name in {"dataset", "indices"}:
                            super().__setattr__(name, value)
                        else:
                            # try to set on dataset if it exists
                            if hasattr(self.dataset, name):
                                setattr(self.dataset, name, value)
                            else:
                                super().__setattr__(name, value)
                
                def dataset_prune_sizes(dataset):
                    targets = dataset.targets
                    utargets, counts = np.unique(targets, return_counts=True)
                    min_count = min(counts)
                    masks = [targets==i for i in utargets]
                    idxs = np.arange(len(targets))
                    idxs = [idxs[m] for m in masks]
                    idxs = [idxs[i][:min_count] for i in range(len(idxs))]
                    idxs = np.concatenate(idxs)
                    dataset = SubsetProxy(dataset, idxs)
                    return dataset
                train_data = dataset_prune_sizes(train_data)
                
            if args.weighted_loss:
                labels = train_data.targets if isinstance(train_data.targets, torch.Tensor) else torch.tensor(train_data.targets)
                counts = torch.bincount(labels)
                class_weights = 1.0 / counts.float()
                class_weights = class_weights / class_weights.sum()  # normalize if needed
            else:
                num_class = len(train_data.classes) if args.dataset != "ImageNet" else args.class_num
                class_weights = torch.ones(num_class)
            class_weights = class_weights.cuda()

            test_data   = utils.Imagenet(root=args.data + '/test',  transform=test_transform,  target_transform=target_transform, class_to_idx=class_to_idx)
            val_data    = utils.Imagenet(root=args.data + '/val',   transform=test_transform,  target_transform=target_transform, class_to_idx=class_to_idx)

        train_loader = DataLoader(train_data, batch_size=tr_bs, num_workers=tr_nw, prefetch_factor=tr_pf, shuffle=True, pin_memory=True, 
            drop_last=True, persistent_workers=tr_pw)
        test_loader = DataLoader(test_data, batch_size=te_bs, num_workers=te_nw, prefetch_factor=te_pf, shuffle=True, 
            pin_memory=True, persistent_workers=te_pw)
        val_loader = DataLoader(val_data, batch_size=te_bs, num_workers=te_nw, prefetch_factor=te_pf, shuffle=True, 
            pin_memory=True, persistent_workers=te_pw)

    num_class = len(train_data.classes) if args.dataset != "ImageNet" else args.class_num
    
    class FeatureQueue: # to make checkpoint load happy
        def __init__(self, *args, **kwargs): pass
        
    model = Net(num_class=num_class, pretrained_path=model_path, image_class=image_class, args=args).cuda()
    model = nn.DataParallel(model)

    optimizer = optim.Adam(model.module.fc.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    results = {'train_loss': [], 'train_acc@1': [], 'train_acc@5': [],
               'test_loss': [], 'test_acc@1': [], 'test_acc@5': []}
    if args.dataset == 'ImageNet':
        results.update({'val_loss': [], 'val_acc@1': [], 'val_acc@5': []})

    # optionally resume from a checkpoint
    best_acc1 = 0
    best_epoch = 0
    resumed = False
    if args.resume:
        if os.path.isfile(args.resume):
            (model, optimizer,
             args.start_epoch, best_acc1, best_epoch,
            ) = load_checkpoint(args.resume, model, optimizer)
             # use current LR, not the one from checkpoint
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr
            resumed = True
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.evaluate:
        epoch = epochs
        train_loss, train_acc_1, train_acc_5 = train_val(model, train_loader, None, tr_bs, args, dataset="train")
        test_loss, test_acc_1, test_acc_5 = train_val(model, test_loader, None, te_bs, args, dataset="test")
        if args.dataset == 'ImageNet':
            val_loss, val_acc_1, val_acc_5 = train_val(model, val_loader, None, te_bs, args, dataset="val")
        if args.txt:
            txt_write = open("{}/{}".format(save_dir, 'result.txt'), 'a')
            txt_write.write('\ntest_loss: {}, test_acc@1: {}, test_acc@5: {}'.format(test_loss, test_acc_1, test_acc_5))
            if args.dataset == 'ImageNet':
                txt_write.write('\nval_loss: {}, val_acc@1: {}, val_acc@5: {}'.format(val_loss, val_acc_1, val_acc_5))
    
    else:
        for epoch in range(1, epochs + 1):
            train_loss, train_acc_1, train_acc_5 = train_val(model, train_loader, optimizer, tr_bs, args, dataset="train")
            if args.val_freq and ((epoch % args.val_freq == 0) or (epoch == epochs)) and (args.dataset == 'ImageNet'):
                val_loss, val_acc_1, val_acc_5 = train_val(model, val_loader, None, te_bs, args, dataset="val")
                is_best = val_acc_1 > best_acc1
                if is_best:
                    best_acc1 = val_acc_1
                    best_epoch = epoch
            else:
                is_best = False
                val_loss = None
            if args.test_freq and ((epoch % args.test_freq == 0) or (epoch == epochs)): # eval knn every test_freq epochs
                test_loss, test_acc_1, test_acc_5 = train_val(model, test_loader, None, te_bs, args, dataset="test")
            else:
                test_loss = None

            if (args.txt) and ((val_loss is not None) or (test_loss is not None)):
                txt_write = open("{}/{}".format(save_dir, 'result.txt'), 'a')
                if test_loss is not None:
                    txt_write.write('\ntest_loss: {}, test_acc@1: {}, test_acc@5: {}'.format(test_loss, test_acc_1, test_acc_5))
                if val_loss is not None:
                    txt_write.write('\nval_loss: {}, val_acc@1: {}, val_acc@5: {}'.format(val_loss, val_acc_1, val_acc_5))

            if (epoch % args.checkpoint_freq == 0) or (epoch == epochs):
                cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

                save_checkpoint({
                    'epoch':                epoch,
                    'state_dict':           model.state_dict(),
                    'best_acc1':            best_acc1,
                    'best_epoch':           best_epoch,
                    'optimizer':            optimizer.state_dict(),
                    "rng_dict": {
                        "rng_state": torch.get_rng_state(),
                        "cuda_rng_state": cuda_rng_state,
                        "numpy_rng_state": np.random.get_state(),
                        "python_rng_state": random.getstate(),
                    },
                }, is_best, args, filename='{}/checkpoint.pth.tar'.format(save_dir))
                              
        # end for epoch in range(1, epochs + 1):
        torch.save(model.state_dict(), '{}/model_{}.pth'.format(save_dir, epoch))
