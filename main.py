import argparse
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
import random
import shutil
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from model import Model
from prepare import prepare_datasets, traverse_objects

def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0

    negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask


def train(net, data_loader, train_optimizer, temperature, debiased, tau_plus, batch_size, args):
    net.train()
    
    transform = data_loader.dataset.transform
    target_transform = data_loader.dataset.target_transform
    
    gradients_batch_size = args.gradients_batch_size
    loader_batch_size = batch_size
    gpu_batch_size = args.micro_batch_size
    
    loader_accum_steps = gradients_batch_size // loader_batch_size 
    gpu_accum_steps = loader_batch_size // gpu_batch_size 
    
    loader_step = 0
    total_samples = len(data_loader.dataset)
    
    total_loss, total_num = 0.0, 0
    bar_format = '{l_bar}{bar:' + str(args.bar) + '}{r_bar}' #{bar:-' + str(args.bar) + 'b}'
    train_bar = tqdm(data_loader,
            total=len(data_loader),
            ncols=args.ncols,               # total width available
            dynamic_ncols=False,            # disable autosizing
            bar_format=bar_format,          # request bar width
            )
    
    train_optimizer.zero_grad()

    for pos_1, pos_2, target, idx in train_bar:
        
        # Split into micro-batches
        pos_1_chunks = pos_1.chunk(gpu_accum_steps)
        pos_2_chunks = pos_2.chunk(gpu_accum_steps)
        target_chunks = target.chunk(gpu_accum_steps)
        idx_chunks = idx.chunk(gpu_accum_steps)

        for pos_1_chunk, pos_2_chunk, target_chunk, ids_chunk in zip(pos_1_chunks, pos_2_chunks, target_chunks, idx_chunks):
            pos_1, pos_2, target = pos_1_chunk.cuda(non_blocking=True), pos_2_chunk.cuda(non_blocking=True), target_chunk
            
            if transform is not None:
                pos_1 = transform(pos_1)
                pos_2 = transform(pos_2)
            if target_transform is not None:
                target = target_transform(target)
                
            feature_1, out_1 = net(pos_1)
            feature_2, out_2 = net(pos_2)

            # neg score
            out = torch.cat([out_1, out_2], dim=0)
            neg = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
            mask = get_negative_mask(args.micro_batch_size).cuda(non_blocking=True)
            neg = neg.masked_select(mask).view(2 * args.micro_batch_size, -1)

            # pos score
            pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
            pos = torch.cat([pos, pos], dim=0)

            # estimator g()
            if debiased:
                N = batch_size * 2 - 2
                Ng = (-tau_plus * N * pos + neg.sum(dim = -1)) / (1 - tau_plus)
                # constrain (optional)
                Ng = torch.clamp(Ng, min = N * np.e**(-1 / temperature))
            else:
                Ng = neg.sum(dim=-1)

            # contrastive loss
            loss = (- torch.log(pos / (pos + Ng) )).mean()
            loss = loss / gpu_accum_steps / loader_accum_steps  # scale loss to account for accumulation

            loss.backward()

            total_num += pos_1_chunk.size(0)
            total_loss += loss.item() * pos_1_chunk.size(0)

            # free memory of micro-batch
            del pos_1_chunk, pos_2_chunk, target_chunk, idx_chunk, loss
            torch.cuda.empty_cache()

        loader_step += 1
        if (loader_step * loader_batch_size) == gradients_batch_size:
            train_optimizer.step()
            loader_step = 0
            train_optimizer.zero_grad()  # clear gradients at beginning of next gradients batch

        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    return total_loss / total_num


# ssl training with IP-IRM
def train_env(net, data_loader, train_optimizer, temperature, updated_split, batch_size, args):
    net.train()

    transform = data_loader.dataset.transform
    target_transform = data_loader.dataset.target_transform

    gradients_batch_size = args.gradients_batch_size
    loader_batch_size = batch_size
    gpu_batch_size = args.micro_batch_size
    
    loader_accum_steps = gradients_batch_size // loader_batch_size 
    gpu_accum_steps = loader_batch_size // gpu_batch_size 
    
    loader_step = 0
    total_samples = len(data_loader.dataset)
    
    total_loss, total_num = 0.0, 0
    bar_format = '{l_bar}{bar:' + str(args.bar) + '}{r_bar}' #{bar:-' + str(args.bar) + 'b}'
    train_bar = tqdm(data_loader,
            total=len(data_loader),
            ncols=args.ncols,               # total width available
            dynamic_ncols=False,            # disable autosizing
            bar_format=bar_format,          # request bar width
            )

    train_optimizer.zero_grad()  # clear gradients at the beginning
        
    for batch_index, data_env in enumerate(train_bar):
        # extract all feature
        pos_1_all, pos_2_all, indexs = data_env[0], data_env[1], data_env[-1]

        # Split into micro-batches
        pos_1_all_chunks = pos_1_all.chunk(gpu_accum_steps)
        pos_2_all_chunks = pos_2_all.chunk(gpu_accum_steps)
        indexs_chunks = indexs.chunk(gpu_accum_steps)
        
        for pos_1_all_chunk, pos_2_all_chunk, indexs_chunk in zip(pos_1_all_chunks, pos_2_all_chunks, indexs_chunks):
        
            pos_1_all_chunk = pos_1_all_chunk.cuda(non_blocking=True)
            pos_2_all_chunk = pos_2_all_chunk.cuda(non_blocking=True)

            if transform is not None:
                pos_1_all_chunk = transform(pos_1_all_chunk)
                pos_2_all_chunk = transform(pos_2_all_chunk)
                
            feature_1_all, out_1_all = net(pos_1_all_chunk)
            feature_2_all, out_2_all = net(pos_2_all_chunk)

            if args.keep_cont: # global contrastive loss (1st partition)
                logits_all, labels_all = utils.info_nce_loss(torch.cat([out_1_all, out_2_all], dim=0), out_1_all.size(0), temperature=temperature)
                loss_original = torch.nn.CrossEntropyLoss()(logits_all, labels_all)

            env_contrastive, env_penalty = [], []

            if isinstance(updated_split, list): # if retain previous partitions
                assert args.retain_group
                for updated_split_each in updated_split:
                    for env in range(args.env_num):

                        out_1, out_2 = utils.assign_features(out_1_all, out_2_all, indexs_chunk, updated_split_each, env)
                        # contrastive loss
                        logits, labels = utils.info_nce_loss(torch.cat([out_1, out_2], dim=0), out_1.size(0), temperature=1.0)
                        logits_cont = logits / temperature

                        loss = torch.nn.CrossEntropyLoss()(logits_cont, labels)
                        # penalty
                        logits_pen = logits / args.irm_temp
                        penalty_score = utils.penalty(logits_pen, labels, torch.nn.CrossEntropyLoss(), mode=args.ours_mode)

                        # collect it into env dict
                        env_contrastive.append(loss)
                        env_penalty.append(penalty_score)

            else:
                for env in range(args.env_num):

                    out_1, out_2 = utils.assign_features(out_1_all, out_2_all, indexs_chunk, updated_split, env)

                    # contrastive loss
                    logits, labels = utils.info_nce_loss(torch.cat([out_1, out_2], dim=0), out_1.size(0), temperature=1.0)
                    logits_cont = logits / temperature
                    logits_pen = logits / args.irm_temp

                    loss = torch.nn.CrossEntropyLoss()(logits_cont, labels)
                    # penalty
                    penalty_score = utils.penalty(logits_pen, labels, torch.nn.CrossEntropyLoss(), mode=args.ours_mode)

                    # collect it into env dict
                    env_contrastive.append(loss)
                    env_penalty.append(penalty_score)

            loss_cont = torch.stack(env_contrastive).mean()
            if args.keep_cont:
                loss_cont += loss_original

            if args.increasing_weight:
                penalty_weight = utils.increasing_weight(0, args.penalty_weight, epoch, args.epochs)
            elif args.penalty_iters < 200:
                penalty_weight = args.penalty_weight if epoch >= args.penalty_iters else 0.
            else:
                penalty_weight = args.penalty_weight
            irm_penalty = torch.stack(env_penalty).mean()
            loss_penalty = irm_penalty
            loss = loss_cont + penalty_weight * loss_penalty

            if penalty_weight > 1.0:
                # Rescale the entire loss to keep gradients in a reasonable range
                loss /= penalty_weight
                
            loss = loss / gpu_accum_steps / loader_accum_steps  # scale loss to account for accumulation

            loss.backward() # adds gradients to accumulated ones
            
            total_num += pos_1_all_chunk.size(0)
            total_loss += loss.item() * pos_1_all_chunk.size(0)

            # free memory of micro-batch
            del pos_1_all_chunk, pos_2_all_chunk, indexs_chunk, loss
            torch.cuda.empty_cache()
        
        # end for pos_1_all_chunk, pos_2_all_chunk, indexs_chunk in zip(pos_1_all_chunks, pos_2_all_chunks, indexs_chunks):

        loader_step += 1
        if (loader_step * loader_batch_size) == gradients_batch_size:
            train_optimizer.step()
            loader_step = 0
            train_optimizer.zero_grad()  # clear gradients at beginning of next gradients batch

        train_bar.set_description('Train Epoch: [{}/{}] [{trained_samples}/{total_samples}]  Loss: {:.4f}  LR: {:.4f}  PW {:.4f}'
            .format(epoch, epochs, total_loss/total_num, train_optimizer.param_groups[0]['lr'], penalty_weight,
            trained_samples=batch_index * batch_size + len(pos_1_all),
            total_samples=len(data_loader.dataset)))

        if batch_index % 10 == 0:
            utils.write_log('Train Epoch: [{:d}/{:d}] [{:d}/{:d}]  Loss: {:.4f}  LR: {:.4f}  PW {:.4f}'
                            .format(epoch, epochs, batch_index * batch_size + len(pos_1_all), len(data_loader.dataset), total_loss/total_num,
                                    train_optimizer.param_groups[0]['lr'], penalty_weight), log_file=log_file)
    # end for batch_index, data_env in enumerate(train_bar):

    return total_loss / total_num



def train_update_split(net, update_loader, soft_split, random_init=False, args=None):
    utils.write_log('Start Maximizing ...', log_file, print_=True)
    
    transform = update_loader.dataset.transform
    target_transform = update_loader.dataset.target_transform
    
    if random_init:
        utils.write_log('Give a Random Split:', log_file, print_=True)
        soft_split = torch.randn(soft_split.size(), requires_grad=True, device="cuda")
        utils.write_log('%s' %(utils.pretty_tensor_str(soft_split[:3])), log_file, print_=True)
    else:
        utils.write_log('Use Previous Split:', log_file, print_=True)
        soft_split = soft_split.requires_grad_()
        utils.write_log('%s' %(utils.pretty_tensor_str(soft_split[:3])), log_file, print_=True)

    if args.offline: # Maximize Step offline, first extract image features
        net.eval()
        feature_bank_1, feature_bank_2 = [], []
        with torch.no_grad():
            # generate feature bank
            bar_format = '{l_bar}{bar:' + str(args.bar) + '}{r_bar}' #{bar:-' + str(args.bar) + 'b}'
            train_bar = tqdm(update_loader_offline,
                total=len(update_loader_offline),
                ncols=args.ncols,               # total width available
                dynamic_ncols=False,            # disable autosizing
                bar_format=bar_format,          # request bar width
                desc='train_update_split(): Feature extracting'
            )
            for pos_1, pos_2, target, Index in train_bar:
                pos_1 = pos_1.cuda(non_blocking=True)
                pos_2 = pos_2.cuda(non_blocking=True)
      
                if transform is not None:
                    pos_1 = transform(pos_1)
                    pos_2 = transform(pos_2)
                if target_transform is not None:
                    target = target_transform(target)
                
                feature_1, out_1 = net(pos_1)
                feature_2, out_2 = net(pos_2)
                feature_bank_1.append(out_1.cpu())
                feature_bank_2.append(out_2.cpu())
        feature1 = torch.cat(feature_bank_1, 0)
        feature2 = torch.cat(feature_bank_2, 0)
        updated_split = utils.auto_split_offline(feature1, feature2, soft_split, temperature, args.irm_temp, loss_mode='v2', irm_mode=args.irm_mode,
                                         irm_weight=args.irm_weight_maxim, constrain=args.constrain, cons_relax=args.constrain_relax, nonorm=args.nonorm, 
                                         log_file=log_file, batch_size=u_bs, num_workers=u_nw, prefetch_factor=u_pf, persistent_workers=u_pw)
    else:
        updated_split = utils.auto_split(net, update_loader, soft_split, temperature, args.irm_temp, loss_mode='v2', irm_mode=args.irm_mode,
                                     irm_weight=args.irm_weight_maxim, constrain=args.constrain, cons_relax=args.constrain_relax, 
                                     nonorm=args.nonorm, log_file=log_file)
    np.save("results/{}/{}/{}_{}{}".format(args.dataset, args.name, 'GroupResults', epoch, ".txt"), updated_split.cpu().numpy())
    return updated_split

def get_feature_bank(net, memory_data_loader, args, progress=False, prefix="Test:"):
    net.eval()
    
    transform = memory_data_loader.dataset.transform
    feature_bank = []
    
    with torch.no_grad():
        # generate feature bank
        bar_format = '{l_bar}{bar:' + str(args.bar) + '}{r_bar}' #{bar:-' + str(args.bar) + 'b}'
        if progress:
            feature_bar = tqdm(memory_data_loader,
                total=len(memory_data_loader),
                ncols=args.ncols,               # total width available
                dynamic_ncols=False,            # disable autosizing
                bar_format=bar_format,          # request bar width
                desc='test(), memory: Feature extracting'
            )
        else:
            feature_bar = memory_data_loader
        for data, _, _ in feature_bar:
            data = data.cuda(non_blocking=True)

            if transform is not None:
                data = transform(data)
                
            feature, out = net(data)
            feature_bank.append(feature)
        #end for data, _, _ in feature_bar:

        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous() # places feature_bank on cuda
        # [N]
        dataset = memory_data_loader.dataset
        if hasattr(dataset, "labels"):
            labels = dataset.labels
        else:
            if dataset.target_transform is not None:
                labels = [dataset.target_transform(t) for t in dataset.targets]
            else:
                labels = dataset.targets        
        feature_labels = torch.tensor(labels, device=feature_bank.device)

    return feature_bank, feature_labels
    
# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def test(net, feature_bank, feature_labels, test_data_loader, args, progress=False, prefix="Test:"):
    net.eval()
       
    total_top1, total_top5, total_num = 0.0, 0.0, 0
    with torch.no_grad():
        # loop test data to predict the label by weighted knn search
        bar_format = '{l_bar}{bar:' + str(args.bar) + '}{r_bar}' #{bar:-' + str(args.bar) + 'b}'
        
        if progress:
            test_bar = tqdm(test_data_loader,
                total=len(test_data_loader),
                ncols=args.ncols,               # total width available
                dynamic_ncols=False,            # disable autosizing
                bar_format=bar_format,          # request bar width
                desc='test(), test: Feature extracting'
            )
        else:
           test_bar = test_data_loader
    
        transform = test_data_loader.dataset.transform
        target_transform = test_data_loader.dataset.target_transform
    
        if args.extract_features:
            test_data_loader.dataset.target_transform = None

        feature_list = []
        pred_labels_list = []
        pred_scores_list = []
        target_list = []
        target_raw_list = []

        for data, _, target in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            
            if transform is not None:
                data = transform(data)

            target_raw = target
            if args.extract_features and target_transform is not None:
                target = target_transform(target_raw).cuda(non_blocking=True)

            feature, out = net(data)

            total_num += data.size(0)
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(feature, feature_bank) # places sim_matrix on cuda
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
            # [B, K]
            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / temperature).exp()

            # counts for each class
            one_hot_label = torch.zeros(data.size(0) * k, c, device=sim_labels.device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1).long(), value=1.0)
            # weighted score ---> [B, C]
            pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, c) * sim_weight.unsqueeze(dim=-1), dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            if progress:
                test_bar.set_description('KNN {} Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'
                                         .format(prefix, epoch, epochs, total_top1 / total_num * 100, total_top5 / total_num * 100))

            # compute output
            if args.extract_features:
                feature_list.append(feature)
                target_list.append(target)
                target_raw_list.append(target_raw)
                pred_labels_list.append(pred_labels)
                pred_scores_list.append(pred_scores)

        # end for data, _, target in test_bar

        if feature_list:
            feature = torch.cat(feature_list, dim=0)
            target = torch.cat(target_list, dim=0)
            target_raw = torch.cat(target_raw_list, dim=0)
            pred_labels = torch.cat(pred_labels_list, dim=0)
            pred_scores = torch.cat(pred_scores_list, dim=0)

            # Save to file
            prefix = "test" if "Test" in prefix else "val"
            directory = f'results/{args.dataset}/{args.name}'
            fp = os.path.join(directory, f"{prefix}_features_dump.pt")       
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

    return total_top1 / total_num * 100, total_top5 / total_num * 100

def save_checkpoint(state, is_best, args, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '{}/{}/model_best.pth.tar'.format(args.save_root, args.name))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
    parser.add_argument('--tau_plus', default=0.1, type=float, help='Positive class priorx')
    parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--dl_tr', default=[256, 4, 2, True], nargs=4, type=str,
                        action=utils.ParseMixed, types=[int, int, int, bool],
                        metavar='DataLoader pars [batch_size, number_workers, prefetch_factor, persistent_workers]', help='Training minimization DataLoader pars')
    parser.add_argument('--dl_u', default=[3096, 4, 2, 1], nargs=4, type=str,
                        action=utils.ParseMixed, types=[int, int, int, bool],
                        metavar='DataLoader pars [batch_size, number_workers, prefetch_factor, persistent_workers]', help='Training Maximization DataLoader pars')
    parser.add_argument('--dl_te', default=[3096, 4, 2, 1], nargs=4, type=str,
                        action=utils.ParseMixed, types=[int, int, int, bool],
                        metavar='DataLoader pars [batch_size, number_workers, prefetch_factor, persistent_workers]', help='Testing/Validation/Memory DataLoader pars')
    parser.add_argument('--micro_batch_size', default=32, type=int, help='batch size on gpu')
    parser.add_argument('--gradients_batch_size', default=256, type=int, help='batch size of gradients accumulation')
    parser.add_argument('--epochs', default=200, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--debiased', default=False, type=bool, help='Debiased contrastive loss or standard loss')
    parser.add_argument('--dataset', type=str, default='STL', choices=['STL', 'CIFAR10', 'CIFAR100', 'ImageNet'], help='experiment dataset')
    parser.add_argument('--data', metavar='DIR', help='path to dataset')
    parser.add_argument('--name', type=str, default='None', help='experiment name')
    parser.add_argument('--pretrain_model', default=None, type=str, help='pretrain model used?')
    parser.add_argument('--baseline', action="store_true", default=False, help='SSL baseline?')
    parser.add_argument('--train_envs', type=str, nargs='+', default=None, required=True)
    parser.add_argument('--test_envs', type=str, nargs='+', default=None)
    parser.add_argument('--holdout_fraction', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=0)

    #### ours model param ####
    parser.add_argument('--ours_mode', default='w', type=str, help='what mode to use')
    parser.add_argument('--penalty_weight', default=1, type=float, help='penalty weight')
    parser.add_argument('--penalty_iters', default=0, type=int, help='penalty weight start iteration')
    parser.add_argument('--increasing_weight', action="store_true", default=False, help='increasing the penalty weight?')
    parser.add_argument('--env_num', default=2, type=int, help='num of the environments')

    parser.add_argument('--maximize_iter', default=30, type=int, help='when maximize iteration')
    parser.add_argument('--irm_mode', default='v1', type=str, help='irm mode when maximizing')
    parser.add_argument('--irm_weight_maxim', default=1, type=float, help='irm weight in maximizing')
    parser.add_argument('--irm_temp', default=0.5, type=float, help='irm loss temperature')
    parser.add_argument('--random_init', action="store_true", default=False, help='random initialization before every time update?')
    parser.add_argument('--constrain', action="store_true", default=False, help='make num of 2 group samples similar?')
    parser.add_argument('--constrain_relax', action="store_true", default=False, help='relax the constrain?')
    parser.add_argument('--retain_group', action="store_true", default=False, help='retain the previous group assignments?')
    parser.add_argument('--debug', action="store_true", default=False, help='debug?')
    parser.add_argument('--nonorm', action="store_true", default=False, help='not use norm for contrastive loss when maximizing')
    parser.add_argument('--groupnorm', action="store_true", default=False, help='use group contrastive loss?')
    parser.add_argument('--offline', action="store_true", default=False, help='save feature at the beginning of the maximize?')
    parser.add_argument('--keep_cont', action="store_true", default=False, help='keep original contrastive?')
    parser.add_argument('--pretrain_path', type=str, default=None, help='the path of pretrain model')

    # image
    parser.add_argument('--image_size', type=int, default=224, help='image size')
    # color in label
    parser.add_argument('--target_transform', type=str, default=None, help='a function definition to apply to target')
    parser.add_argument('--class_to_idx', type=str, default=None, help='a function definition to apply to class to obtain it index')
    parser.add_argument('--image_class', choices=['ImageNet', 'STL', 'CIFAR'], default='ImageNet', help='Image class, default=ImageNet')
    parser.add_argument('--class_num', default=1000, type=int, help='num of classes')
    parser.add_argument('--ncols', default=80, type=int, help='number of columns in terminal')
    parser.add_argument('--bar', default=50, type=int, help='length of progess bar')

    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
    parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
    parser.add_argument('--save_root', type=str, default='save', help='root dir for saving')
    parser.add_argument('--checkpoint_freq', default=3, type=int, metavar='N',
                    help='checkpoint epoch freqeuncy')
    parser.add_argument('--val_freq', default=3*3, type=int, metavar='N',
                    help='validation epoch freqeuncy')
    parser.add_argument('--test_freq', default=5*5, type=int, metavar='N',
                    help='test epoch freqeuncy')
    parser.add_argument('--norandgray', action="store_true", default=False, help='skip rand gray transform')
    parser.add_argument('--evaluate', action="store_true", default=False, help='only evaluate')
    parser.add_argument('--extract_features', action="store_true", help="extract features for post processiin during evaluate")

    # args parse
    args = parser.parse_args()

    # seed
    utils.set_seed(args.seed)

    feature_dim, temperature, tau_plus, k = args.feature_dim, args.temperature, args.tau_plus, args.k
    epochs, debiased,  = args.epochs,  args.debiased
    dl_tr, dl_te, dl_u = args.dl_tr, args.dl_te, args.dl_u
    target_transform = eval(args.target_transform) if args.target_transform is not None else None
    class_to_idx = eval(args.class_to_idx) if args.class_to_idx is not None else None
    image_class, image_size = args.image_class, args.image_size

    if not os.path.exists('results/{}/{}'.format(args.dataset, args.name)):
        os.makedirs('results/{}/{}'.format(args.dataset, args.name))
    log_file = 'results/{}/{}/log.txt'.format(args.dataset, args.name)
    if not os.path.exists('{}/{}'.format(args.save_root, args.name)):
        os.makedirs('{}/{}'.format(args.save_root, args.name))

    # data prepare
    tr_bs, tr_nw, tr_pf, tr_pw = dl_tr
    te_bs, te_nw, te_pf, te_pw = dl_te
    u_bs, u_nw, u_pf, u_pw = dl_u
    if args.dataset == 'STL':
        train_transform = utils.make_train_transform(normalize=args.image_class)
        test_transform = utils.make_test_transform(normalize=args.image_class)
        train_data = utils.STL10Pair_Index(root=args.data, split='train+unlabeled', transform=train_transform)
        update_data = utils.STL10Pair_Index(root=args.data, split='train+unlabeled', transform=train_transform, target_transform=target_transform)
        memory_data = utils.STL10Pair(root=args.data, split='train', transform=test_transform, target_transform=target_transform)
        test_data = utils.STL10Pair(root=args.data, split='test', transform=test_transform, target_transform=target_transform)
        train_loader = DataLoader(train_data, batch_size=tr_bs, num_workers=tr_nw, prefetch_factor=tr_pf, shuffle=True, pin_memory=True, 
            drop_last=True, persistent_workers=tr_pw)
        update_loader = DataLoader(update_data, batch_size=u_bs, num_workers=u_nw, prefetch_factor=u_pf, shuffle=True, pin_memory=True, 
            drop_last=True, persistent_workers=u_pw)
        update_loader_offline = DataLoader(update_data, batch_size=u_bs, num_workers=u_nw, prefetch_factor=u_pf, shuffle=False, 
            pin_memory=True, persistent_workers=u_pw)
        memory_loader = DataLoader(memory_data, batch_size=te_bs, num_workers=te_nw, prefetch_factor=te_pf, shuffle=False, 
            pin_memory=True, persistent_workers=te_pw)
        test_loader = DataLoader(test_data, batch_size=te_bs, num_workers=te_nw, prefetch_factor=te_pf, shuffle=False, 
            pin_memory=True, persistent_workers=te_pw)
    elif args.dataset == 'CIFAR10':
        train_transform = utils.make_train_transform(normalize=args.image_class)
        test_transform = utils.make_test_transform(normalize=args.image_class)
        train_data = utils.CIFAR10Pair_Index(root=args.data, train=True, transform=train_transform, target_transform=target_transform, download=True)
        update_data = utils.CIFAR10Pair_Index(root=args.data, train=True, transform=train_transform, target_transform=target_transform)
        memory_data = utils.CIFAR10Pair(root=args.data, train=True, transform=test_transform, target_transform=target_transform)
        test_data = utils.CIFAR10Pair(root=args.data, train=False, transform=test_transform, target_transform=target_transform)
        train_loader = DataLoader(train_data, batch_size=tr_bs, num_workers=tr_nw, prefetch_factor=tr_pf, shuffle=True, pin_memory=True, 
            drop_last=True, persistent_workers=tr_pw)
        update_loader = DataLoader(update_data, batch_size=u_bs, num_workers=u_nw, prefetch_factor=u_pf, shuffle=True, pin_memory=True, 
            drop_last=True, persistent_workers=u_pw)
        update_loader_offline = DataLoader(update_data, batch_size=u_bs, num_workers=u_nw, prefetch_factor=u_pf, shuffle=False, 
            pin_memory=True, persistent_workers=u_pw)
        memory_loader = DataLoader(memory_data, batch_size=te_bs, num_workers=te_nw, prefetch_factor=te_pf, shuffle=False, 
            pin_memory=True, persistent_workers=te_pw)
        test_loader = DataLoader(test_data, batch_size=te_bs, num_workers=te_nw, prefetch_factor=te_pf, shuffle=False, 
            pin_memory=True, persistent_workers=te_pw)
    elif args.dataset == 'CIFAR100':
        train_transform = utils.make_train_transform(normalize=args.image_class)
        test_transform = utils.make_test_transform(normalize=args.image_class)
        train_data = utils.CIFAR100Pair_Index(root=args.data, train=True, transform=train_transform, target_transform=target_transform)
        update_data = utils.CIFAR100Pair_Index(root=args.data, train=True, transform=train_transform, target_transform=target_transform)
        memory_data = utils.CIFAR100Pair(root=args.data, train=True, transform=test_transform, target_transform=target_transform)
        test_data = utils.CIFAR100Pair(root=args.data, train=False, transform=test_transform, target_transform=target_transform)
        train_loader = DataLoader(train_data, batch_size=tr_bs, num_workers=tr_nw, prefetch_factor=tr_pf, shuffle=True, pin_memory=True, 
            drop_last=True, persistent_workers=tr_pw)
        update_loader = DataLoader(update_data, batch_size=u_bs, num_workers=u_nw, prefetch_factor=u_pf, shuffle=True, pin_memory=True, 
            drop_last=True, persistent_workers=u_pw)
        update_loader_offline = DataLoader(update_data, batch_size=u_bs, num_workers=u_nw, prefetch_factor=u_pf, shuffle=False, 
            pin_memory=True, persistent_workers=u_pw)
        memory_loader = DataLoader(memory_data, batch_size=te_bs, num_workers=te_nw, prefetch_factor=te_pf, shuffle=False, 
            pin_memory=True, persistent_workers=te_pw)
        test_loader = DataLoader(test_data, batch_size=te_bs, num_workers=te_nw, prefetch_factor=te_pf, shuffle=False, 
            pin_memory=True, persistent_workers=te_pw)
    elif args.dataset == 'ImageNet':
        train_transform = utils.make_train_transform(image_size, randgray=not args.norandgray, normalize=args.image_class)
        test_transform = utils.make_test_transform(normalize=args.image_class)
        wrap = args.extract_features
        # descriptors of train data
        train_desc  =   {'dataset': utils.Imagenet_idx_pair,
                          'transform': train_transform,
                          'target_transform': target_transform,
                          'class_to_index': class_to_idx,
                          'wrap': False, # for changeable target transform
                          'target_pos': 2,
                          'required_split': "in",
                        }
        update_desc =   {'dataset': utils.Imagenet_idx_pair,
                          'transform': train_transform,
                          'target_transform': target_transform,
                          'class_to_index': class_to_idx,
                          'wrap': False, # for changeable target transform
                          'target_pos': 2,
                          'required_split': "in",
                        }
        memory_desc =   {'dataset': utils.Imagenet_pair,
                          'transform': test_transform,
                          'target_transform': target_transform,
                          'class_to_index': class_to_idx,
                          'wrap': False, # for changeable target transform
                          'target_pos': 2,
                          'required_split': "in",
                        }
        val_desc    =   {'dataset': utils.Imagenet_pair,
                          'transform': test_transform,
                          'target_transform': target_transform,
                          'class_to_index': class_to_idx,
                          'wrap': wrap, # for changeable target transform
                          'target_pos': 2,
                          'required_split': "out",
                        }
        # descriptors of test data
        test_desc   =   {'dataset': utils.Imagenet_pair,
                          'transform': test_transform,
                          'target_transform': target_transform,
                          'class_to_index': class_to_idx,
                          'wrap': wrap, # for changeable target transform
                          'target_pos': 2,
                          'required_split': "in",
                        }


        datas = prepare_datasets(args.data, args.train_envs, [train_desc, update_desc, memory_desc, val_desc], args.holdout_fraction, args.seed)
        train_data, update_data, memory_data, val_data = tuple(data[0] for data in datas)

        datas = prepare_datasets(args.data, args.test_envs, [test_desc], 1.0, args.seed)
        test_data = datas[0][0]

        #traverse_objects(update_data)
        #exit()
        train_loader = DataLoader(train_data, batch_size=tr_bs, num_workers=tr_nw, prefetch_factor=tr_pf, shuffle=True, pin_memory=True, 
            drop_last=True, persistent_workers=tr_pw)
        update_loader = DataLoader(update_data, batch_size=u_bs, num_workers=u_nw, prefetch_factor=u_pf, shuffle=True, pin_memory=True, 
            drop_last=True, persistent_workers=u_pw)
        update_loader_offline = DataLoader(update_data, batch_size=u_bs, num_workers=u_nw, prefetch_factor=u_pf, shuffle=False, 
            pin_memory=True, persistent_workers=u_pw)
        memory_loader = DataLoader(memory_data, batch_size=te_bs, num_workers=te_nw, prefetch_factor=te_pf, shuffle=False, 
            pin_memory=True, persistent_workers=te_pw)
        test_loader = DataLoader(test_data, batch_size=te_bs, num_workers=te_nw, prefetch_factor=te_pf, shuffle=False, 
            pin_memory=True, persistent_workers=te_pw)
        val_loader = DataLoader(val_data, batch_size=te_bs, num_workers=te_nw, prefetch_factor=te_pf, shuffle=False, 
            pin_memory=True, persistent_workers=te_pw)

    # model setup and optimizer config
    model = Model(feature_dim, image_class=image_class).cuda()
    model = nn.DataParallel(model)
    # pretrain model
    if args.pretrain_path is not None and os.path.isfile(args.pretrain_path):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        msg = []
        print("=> loading pretrained checkpoint '{}'".format(args.pretrain_path), end="")
        checkpoint = torch.load(args.pretrain_path, map_location=device, weights_only=False)
        if 'state_dict' in checkpoint.keys():
            state_dict = checkpoint['state_dict']
            print(f"Epoch: {checkpoint['epoch']}")
        else:
            state_dict = checkpoint
            print("Epoch: N/A")
        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)
    else:
        print('Using default model')


    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    c = len(memory_data.classes) if args.dataset != "ImageNet" else args.class_num
    print('# Classes: {}'.format(c))

    # optionally resume from a checkpoint
    best_acc1 = 0
    best_epoch = 0
    resumed = False
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, weights_only=False)
            args.start_epoch = checkpoint['epoch'] + 1
            best_acc1 = checkpoint['best_acc1']
            best_epoch = checkpoint['best_epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            updated_split = checkpoint['updated_split']
            updated_split_all = checkpoint['updated_split_all']
            # Restore RNG states
            rng_dict = checkpoint['rng_dict']
            torch.set_rng_state(rng_dict['rng_state'].cpu())
            if torch.cuda.is_available():
                torch.cuda.set_rng_state_all([t.cpu() for t in rng_dict['cuda_rng_state']])
            np.random.set_state(rng_dict['numpy_rng_state'])
            random.setstate(rng_dict['python_rng_state'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            resumed = True
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # training loop
    if not os.path.exists('results'):
        os.mkdir('results')

    epoch = args.start_epoch

    if args.evaluate:
        print(f"Staring evaluation name: {args.name}")
        print('eval on val data')
        feauture_bank, feature_labels = get_feature_bank(net, memory_data_loader, args, progress=True, prefix="Evaluate:")
        val_acc_1, val_acc_5 = test(model, feauture_bank, feature_labels, val_loader, args, progress=True, prefix="Val:")
        print('eval on test data')
        test_acc_1, test_acc_5 = test(model, feauture_bank, feature_labels, test_loader, args, progress=True, prefix="Test:")
        exit()

    # update partition for the first time
    if not args.baseline and not resumed:
        if args.dataset != "ImageNet":
            updated_split = torch.randn((len(update_data), args.env_num), requires_grad=True, device="cuda")
        else:
            updated_split = torch.randn((len(update_data), args.env_num), requires_grad=True, device="cuda")
        updated_split = train_update_split(model, update_loader, updated_split, random_init=args.random_init, args=args)
        updated_split_all = [updated_split.clone().detach()]


    for epoch in range(args.start_epoch, epochs + 1):
        if args.baseline:
            train_loss = train(model, train_loader, optimizer, temperature, debiased, tau_plus, tr_bs, args)
        else: # Minimize Step
            if args.retain_group: # retain the previous partitions
                train_loss = train_env(model, train_loader, optimizer, temperature, updated_split_all, tr_bs, args)
            else:
                train_loss = train_env(model, train_loader, optimizer, temperature, updated_split, tr_bs, args)

            if epoch % args.maximize_iter == 0: # Maximize Step
                updated_split = train_update_split(model, update_loader, updated_split, random_init=args.random_init, args=args)
                updated_split_all.append(updated_split)
       
        if (epoch % args.test_freq == 0) or \
           ((epoch % args.val_freq == 0) and (args.dataset == 'ImageNet')) or \
           (epoch == epochs): # eval knn every test_freq/val_freq and last epochs
                feauture_bank, feature_labels = get_feature_bank(model, memory_loader, args, progress=True, prefix="Evaluate:")

        if (epoch % args.test_freq == 0) or (epoch == epochs): # eval knn every test_freq epochs
            test_acc_1, test_acc_5 = test(model, feauture_bank, feature_labels, test_loader, args, progress=True, prefix="Test:")
            txt_write = open("results/{}/{}/{}".format(args.dataset, args.name, 'knn_result.txt'), 'a')
            txt_write.write('\ntest_acc@1: {}, test_acc@5: {}'.format(test_acc_1, test_acc_5))
            torch.save(model.state_dict(), 'results/{}/{}/model_{}.pth'.format(args.dataset, args.name, epoch))

        if ((epoch % args.val_freq == 0) or (epoch == epochs)) and (args.dataset == 'ImageNet'):
            # evaluate on validation set
            acc1, _ = test(model, feauture_bank, feature_labels, val_loader, args, progress=True, prefix="Val:")

            # remember best acc@1 & best epoch and save checkpoint
            is_best = acc1 > best_acc1
            if is_best:
                best_acc1 = acc1
                best_epoch = epoch
        else:
            is_best = False

        if (epoch % args.checkpoint_freq == 0) or (epoch == epochs):
            cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

            save_checkpoint({
                'epoch':                epoch,
                'state_dict':           model.state_dict(),
                'best_acc1':            best_acc1,
                'best_epoch':           best_epoch,
                'optimizer':            optimizer.state_dict(),
                'updated_split':        updated_split,
                'updated_split_all':    updated_split_all,
                "rng_dict": {
                    "rng_state": torch.get_rng_state(),
                    "cuda_rng_state": cuda_rng_state,
                    "numpy_rng_state": np.random.get_state(),
                    "python_rng_state": random.getstate(),
                },
            }, is_best, args, filename='{}/{}/checkpoint.pth.tar'.format(args.save_root, args.name))
