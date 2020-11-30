#!/usr/bin/env python

from __future__ import print_function

import argparse
import glob
import logging
import os
import re
import shutil
import sys
import traceback
import random
import time
import math

import torch

import torch.nn as nn
import horovod.torch as hvd

from utils.pytorch_data import KaldiArkDataset
from utils import ze_utils
from models.metrics import AddMarginProduct, ArcMarginProduct, SphereProduct
from models.resnet2 import *


APEX_AVAILABLE = False

torch.backends.cudnn.benchmark = True

logger = logging.getLogger('train-pytorch')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s [%(module)s:%(lineno)s - %(levelname).1s] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def get_args():
    """ Get args from stdin.
    """

    parser = argparse.ArgumentParser(
        description="""Trains a CNN PyTorch model using segment-level 
        objectives like cross-entropy and mean-squared-error (there is only 
        one output for each input segment using statestics pooling layer).
        DNNs include TDNNs and CNNs and should be defined in the 
        models_pytorch.py file as a separate class.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        conflict_handler='resolve')

    parser.add_argument("--use-gpu", type=str, dest='use_gpu', choices=["yes", "no"],
                        help="Use GPU for training.", default="yes")

    parser.add_argument("--fp16-compression", type=str, dest='fp16_compression', choices=["yes", "no"],
                        help="Use 16 bit flouting point compression.", default="no")

    parser.add_argument("--apply-cmn", type=str, dest='apply_cmn', choices=["yes", "no"],
                        help="Apply one more CMN on training examples.", default="no")

    parser.add_argument("--momentum", type=float, dest='momentum', default=0.0,
                        help="""Momentum used in update computation.""")

    parser.add_argument("--model", type=str, dest='model', required=True,
                        help="Shows the class name which should be defined in models/*.py file.")

    parser.add_argument("--metric", type=str, dest='metric', default="linear",
                        choices=['linear', 'add_margin', 'arc_margin', 'sphere'],
                        help="Shows the custum metric that should be used as last layer.")

    parser.add_argument("--dir", type=str, required=True,
                        help="The main directory for this experiment to store "
                             "the models and all other files such as logs. The"
                             "trained models will saved in models sub-directory.")
    parser.add_argument("--model-init", type=str, default=None,
                        help="Provided pretrained model")

    parser.add_argument("--egs-dir", type=str, dest='egs_dir', required=True,
                        help="Directory of training egs in Kaldi like. Note that we slightly"
                             "changed the Kaldi egs directory and so you should use our"
                             "script to create the egs_dir.")

    parser.add_argument("--num-epochs", type=int, dest='num_epochs', default=3,
                        help="Number of epochs to train the model.")

    parser.add_argument("--num-targets", type=int, dest='num_targets', required=True,
                        help="Shows the number of output of the neural network, here "
                             "number of speakers in the training data.")
    parser.add_argument("--embed-dim", type=int, dest="embed_dim", default=128,
                        help="The dimension of the speaker embeddings")

    parser.add_argument("--initial-effective-lrate", type=float,
                        dest='initial_effective_lrate', default=0.001,
                        help="Learning rate used during the initial iteration.")

    parser.add_argument("--final-effective-lrate", type=float,
                        dest='final_effective_lrate', default=None,
                        help="Learning rate used during the final iteration.")
    parser.add_argument("--initial-margin-m", type=float, default=None,
                        help="hyper parameter margin used for the initial iteration")

    parser.add_argument("--final-margin-m", type=float, default=None,
                        help="hyper parameter margin used for the final iteration")

    parser.add_argument("--optimizer", type=str, dest='optimizer', default='SGD', choices=['SGD', 'Adam'],
                        help="Optimizer for training.")

    parser.add_argument('--warmup-epochs', type=int, dest='warmup_epochs', default=0,
                        help='number of warmup epochs')

    parser.add_argument("--optimizer-weight-decay", type=float, dest='optimizer_weight_decay',
                        default=0, help="Optimizer weight decay for training.")

    parser.add_argument("--minibatch-size", type=int, dest='minibatch_size', required=True,
                        help="Size of the minibatch used in SGD/Adam training.")

    parser.add_argument("--frame-downsampling", type=int, dest='frame_downsampling', default=0,
                        help="For downsampling frames. This shows number of frames to ignore."
                             "Defult is zero and means no downsampling.")

    parser.add_argument("--random-seed", type=int, dest='random_seed', default=0,
                        help="""Sets the random seed for PyTorch random seed.  Note that we don't
                        shuffle examples for reading speed.  The examples was already shuffles once
                        using Kaldi in preparation stage.  Warning: This random seed does not control
                        all aspects of this experiment.  There might be other random seeds used in 
                        other stages of the experiment like data preparation (e.g. volume perturbation).""")

    parser.add_argument("--preserve-model-interval", dest="preserve_model_interval",
                        type=int, default=100, help="""Determines iterations for which 
                        models will be preserved during cleanup.  If mod(iter, preserve_model_interval) == 0
                        model will be preserved.""")

    parser.add_argument("--cleanup", default=False, action='store_true', help="Clean up models after training.")

    parser.add_argument("--stage", type=int, default=-2,
                        help="Specifies the stage of the training to execution from.")

    parser.add_argument('--use-apex', default=False, action='store_true', help='use APEX if available')

    parser.add_argument('--fix-margin-m', default=None, type=int,
                        help='fix margin m parameter after Nth iteration to its final value')
    
    parser.add_argument('--squeeze-excitation', default=False, action='store_true',
                        help='use squeeze excitation layers')

    args = parser.parse_args()

    args = process_args(args)

    # apex

    try:
        if args.use_apex:
            from apex import amp
        else:
            raise ModuleNotFoundError
        APEX_AVAILABLE = True
    except ModuleNotFoundError:
        APEX_AVAILABLE = False

    return args


def process_args(args):
    """ Process the options got from get_args() """
    models_dir = os.path.join(args.dir, 'models')
    log_dir = os.path.join(args.dir, 'log')
    if not os.path.isdir(models_dir):
        os.makedirs(models_dir)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    file_handler = logging.FileHandler(os.path.join(args.dir, 'log', 'train.log'))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(' '.join(sys.argv))
    logger.info(f'Running on host: {str(os.uname()[1])}')

    args.use_gpu = args.use_gpu == 'yes'
    args.fp16_compression = args.fp16_compression == 'yes'
    args.apply_cmn = args.apply_cmn == 'yes'

    return args


def train_one_iteration(args, main_dir, _iter, model, data_loader, optimizer, criterion,
                        device, log_interval, len_train_sampler):
    """ Called from train for one iteration of neural network training

    Selected args:
        shrinkage_value: If value is 1.0, no shrinkage is done; otherwise
            parameter values are scaled by this value.
    """

    iter_loss = 0
    num_correct = 0
    num_total = 0

    start_time = time.time()
    model.train()
    len_data_loader = len(data_loader)
    total_gpu_waiting = 0
    batch_idx = 0
    try:
        for batch_idx, (data, target) in enumerate(data_loader):
            gpu_waiting = time.time()
            target = target.long()

            if args.frame_downsampling > 0:
                ss = random.randint(0, args.frame_downsampling)
                data = data[:, ss::args.frame_downsampling + 1, :]

            data = data.to(torch.device(device=device))
            target = target.to(torch.device(device=device))
            data = data.transpose(1, 2)
            optimizer.zero_grad()
            output = model(data)
            if args.metric == 'linear':
                output = model.metric(output)
            else:
                output = model.metric(output, target)

            loss = criterion(output, target)

            if APEX_AVAILABLE:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                    optimizer.synchronize()
                with optimizer.skip_synchronize():
                    optimizer.step()
            else:
                loss.backward()
                optimizer.step()

            loss = loss.item()
            iter_loss += loss
            predict = output.max(1)[1]
            num_correct += predict.eq(target).sum().item()
            num_total += len(data)
            total_gpu_waiting += time.time() - gpu_waiting
            if batch_idx > 0 and batch_idx % log_interval == 0 and hvd.rank() == 0:
                # use train_sampler to determine the number of examples in this worker's partition.
                logger.info(f'Train Iter: {_iter} [{batch_idx * len(data)}/{len_train_sampler} '
                            f'({100.0 * batch_idx / len_data_loader:.0f}%)]  Loss: {loss:.6f}')
    except RuntimeError as e:
        logger.warning(f'RuntimeError when processing mini batch {batch_idx + 1}/{len_data_loader}. '
                       f'If you see this message frequently, it probably means, '
                       f'that there is some problem with your data or code. {os.linesep}{e}')
    except TypeError as e:
        logger.warning(f'TypeError when processing mini batch {batch_idx + 1}/{len_data_loader}. '
                       f'If you see this message frequently, it probably means, '
                       f'that there is some problem with your data or code. {os.linesep}{e}')
    acc = num_correct * 1.0 / num_total
    iter_loss /= len_data_loader

    # save the model and do logging in worker with rank zero
    args.processed_archives += hvd.size()
    if hvd.rank() == 0:
        logger.info(f'Iteration Loss: {iter_loss:.4f}  Accuracy: {acc * 100:.2f}%')
        logger.info(f'Elapsed time: {(time.time() - start_time) / 60.0:.2f} minutes '
                    f'and GPU waiting: {total_gpu_waiting / 60.0:.2f} minutes.')
        new_model_path = os.path.join(main_dir, 'models', f'raw_{_iter}.pth')
        logger.info(f'Saving model to: {new_model_path}')
        saving_model = model
        while isinstance(saving_model, torch.nn.DataParallel):
            saving_model = saving_model.module
        _save_checkpoint(
            {
                'processed_archives': args.processed_archives,
                'class_name': args.model,
                'frame_downsampling': args.frame_downsampling,
                'state_dict': saving_model.state_dict(),
                'optimizer': optimizer.state_dict(),
            },
            file_name=new_model_path)


def eval_network(model, data_loader, device):
    """ Called from train for one iteration of neural network training

    Selected args:
        shrinkage_value: If value is 1.0, no shrinkage is done; otherwise
            parameter values are scaled by this value.
    """

    iter_loss = 0
    num_correct = 0
    num_total = 0

    start_time = time.time()
    model.eval()
    len_data_loader = len(data_loader)
    total_gpu_waiting = 0
    batch_idx = 0
    try:
        for batch_idx, (data, target) in enumerate(data_loader):
            gpu_waiting = time.time()
            target = target.long()
            data = data.to(torch.device(device=device))
            target = target.to(torch.device(device=device))
            data = data.transpose(1, 2)
            output, _, _ = model(data)

            if args.metric == "linear":
                output = model.metric(output)
            else:
                output = model.metric(output, target)
            loss = model.criterion(output, target) + model.get_l2_loss()
            loss = loss.item()
            iter_loss += loss
            predict = output.max(1)[1]
            num_correct += predict.eq(target).sum().item()
            num_total += len(data)
            total_gpu_waiting += time.time() - gpu_waiting
    except RuntimeError:
        logger.warning(f'RuntimeError when processing mini batch {batch_idx + 1}/{len_data_loader}. '
                       f'If you see this message frequently, it probably means, '
                       f'that there is some problem with your data or code.')
    acc = num_correct * 1.0 / num_total
    iter_loss /= len_data_loader

    # save the model and do loging in worker with rank zero
    logger.info('Final Iteration Loss: {:.6f}\t and Accuracy: {:.2f}%'.format(iter_loss, acc * 100))
    logger.info("Elapsed time for processing whole training minibatches is %.2f minutes." %
                ((time.time() - start_time) / 60.0))
    logger.info("GPU waiting for processing whole training minibatches is %.2f minutes." %
                (total_gpu_waiting / 60.0))


def _remove_model(nnet_dir, _iter, preserve_model_interval=100):
    if _iter < 1 or _iter % preserve_model_interval == 0:
        return
    model_path = os.path.join(nnet_dir, 'models', f'raw_{_iter}.pth')
    if os.path.exists(model_path):
        os.remove(model_path)


def _save_checkpoint(state, file_name, is_best=False):
    torch.save(state, file_name)
    if is_best:
        shutil.copyfile(file_name, 'model_best.pth')


def train(args):
    """ The main function for training.

    Args:
        args: a Namespace object with the required parameters
            obtained from the function process_args()
    """

    egs_dir = args.egs_dir
    # initialize horovod
    hvd.init()

    # verify egs dir and extract parameters
    [num_archives, egs_feat_dim, arks_num_egs] = ze_utils.verify_egs_dir(egs_dir)
    assert arks_num_egs > 0

    device = 'cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu'
    torch.cuda.set_device(hvd.local_rank())

    if hvd.rank() == 0:
        logger.info(f'Device is: {device}')

    num_jobs = hvd.size()
    if hvd.rank() == 0:
        logger.info(f'Using {num_jobs} training jobs.')
    if num_jobs > num_archives:
        raise ValueError('num_jobs cannot exceed the number of archives in the egs directory')

    init_model_path = args.model_init

    # add metric
    try:
        if args.metric == 'add_margin':
            metric_fc = AddMarginProduct(args.embed_dim, args.num_targets, s=32, m=0.2)
        elif args.metric == 'arc_margin':
            metric_fc = ArcMarginProduct(args.embed_dim, args.num_targets, s=32, m=0.2)
        elif args.metric == 'sphere':
            metric_fc = SphereProduct(args.embed_dim, args.num_targets, m=4)
        else:
            metric_fc = nn.Sequential(nn.BatchNorm1d(args.embed_dim),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(args.embed_dim, args.num_targets))

        model = eval(args.model)(feat_dim=egs_feat_dim, embed_dim=args.embed_dim, squeeze_excitation=args.squeeze_excitation)

        # load the init model if exist. This is useful when loading from a pre-trained model
        if init_model_path is not None:
            if hvd.rank() == 0:
                logger.info(f'Loading the initial network from: {init_model_path}')
            checkpoint = torch.load(init_model_path, map_location='cpu')
            model.load_state_dict(checkpoint['state_dict'], strict=False)
    except AttributeError:
        raise AttributeError(f'The specified class name {args.model} does not exist.')

    model.add_module('metric', metric_fc)

    # move model to device before creating optimizer
    model = model.to(torch.device(device=device))

    parameters = model.parameters()

    # scale learning rate by the number of GPUs.
    if args.optimizer == 'SGD':
        main_optimizer = torch.optim.SGD(parameters, lr=args.initial_effective_lrate * num_jobs,
                                         momentum=args.momentum, weight_decay=args.optimizer_weight_decay,
                                         nesterov=True)
    elif args.optimizer == 'Adam':
        main_optimizer = torch.optim.Adam(parameters, lr=args.initial_effective_lrate * num_jobs,
                                          weight_decay=args.optimizer_weight_decay)
    else:
        raise ValueError(f'Invalid optimizer {args.optimizer}.')

    if hvd.rank() == 0:
        logger.info(str(model))
        logger.info(str(main_optimizer))

    processed_archives = 0
    if init_model_path is not None and hvd.rank() == 0:
        logger.info(f'Saving the initial network to: {init_model_path}')
        _save_checkpoint({
            'processed_archives': processed_archives,
            'class_name': args.model,
            'frame_downsampling': args.frame_downsampling,
            'state_dict': model.state_dict(),
            'optimizer': main_optimizer.state_dict(),
        }, file_name=init_model_path)

    # save kaldi's files
    if hvd.rank() == 0:
        with open(os.path.join(args.dir, 'models', 'model_info'), 'wt') as fid:
            fid.write(f'{args.model} {egs_feat_dim} {args.num_targets}{os.linesep}')
        with open(os.path.join(args.dir, 'models', 'config.txt'), 'wt') as fid:
            fid.write(str(model))
        with open(os.path.join(args.dir, 'command.sh'), 'wt') as fid:
            fid.write(' '.join(sys.argv) + '\n')
        with open(os.path.join(args.dir, 'max_chunk_size'), 'wt') as fid:
            fid.write(f'10000{os.linesep}')
        with open(os.path.join(args.dir, 'min_chunk_size'), 'wt') as fid:
            fid.write(f'25{os.linesep}')

    # find the last saved model and load it
    saved_models = glob.glob(os.path.join(args.dir, 'models', 'raw_*.pth'))
    finished_iterations = 0
    try:
        for name in saved_models:
            model_id = int(re.split('[_\.]', name)[-2])
            if model_id > finished_iterations:
                finished_iterations = model_id
    except Exception:
        pass

    if finished_iterations > 0:
        model_path = os.path.join(args.dir, 'models', f'raw_{finished_iterations}.pth')
        if hvd.rank() == 0:
            logger.info('Loading model from ' + model_path)

        checkpoint = torch.load(model_path, map_location='cpu')
        processed_archives = checkpoint['processed_archives']
        model.load_state_dict(checkpoint['state_dict'])
        main_optimizer.load_state_dict(checkpoint['optimizer'])

    # note: here minibatch is the size before
    train_dataset = KaldiArkDataset(egs_dir=egs_dir, num_archives=num_archives, num_workers=num_jobs, rank=hvd.rank(),
                                    num_examples_in_each_ark=arks_num_egs, finished_iterations=finished_iterations,
                                    processed_archives=processed_archives, apply_cmn=args.apply_cmn)

    # use DistributedSampler to partition the training data.
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=num_jobs, rank=hvd.rank())

    kwargs = {'drop_last': False, 'pin_memory': False}
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.minibatch_size, 
        sampler=train_sampler, 
        **kwargs)

    # (optional) compression algorithm. TODO add better compression method
    compression = hvd.Compression.fp16 if args.fp16_compression else hvd.Compression.none
    #
    # wrap optimizer with DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(main_optimizer, named_parameters=model.named_parameters(),
                                         compression=compression)

    # broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    criterion = nn.CrossEntropyLoss()
    args.processed_archives = processed_archives

    # set num_iters so that as close as possible, we process the data
    num_iters = int(args.num_epochs * num_archives * 1.0 / num_jobs)
    num_archives_to_process = int(num_iters * num_jobs)

    # initialize APEX
    if APEX_AVAILABLE:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1', loss_scale='dynamic')

    if hvd.rank() == 0:
        logger.info(f'Training will run for {args.num_epochs} epochs = {num_iters} iterations')

    for _iter in range(finished_iterations, num_iters):
        percent = args.processed_archives * 100.0 / num_archives_to_process
        epoch = (args.processed_archives * float(args.num_epochs) / num_archives_to_process)

        effective_learning_rate = args.initial_effective_lrate

        if args.final_effective_lrate is not None:
            effective_learning_rate = math.exp(args.processed_archives * math.log(args.final_effective_lrate /
                                                                                  args.initial_effective_lrate)
                                               / num_archives_to_process) \
                                      * args.initial_effective_lrate

        if args.metric != 'linear':
            effective_margin_m = args.initial_margin_m
            if args.final_margin_m is not None or args.fix_margin_m is not None:
                if args.fix_margin_m is not None:
                    if epoch > args.fix_margin_m:
                        # keep margin fixed
                        num_archives_to_process_margin_m = args.processed_archives
                    else:
                        num_archives_to_process_margin_m = num_archives_to_process - (num_archives_to_process / args.num_epochs) * (args.num_epochs - args.fix_margin_m)
                else:
                    num_archives_to_process_margin_m = num_archives_to_process 
                effective_margin_m = math.exp(args.processed_archives * math.log(args.final_margin_m / args.initial_margin_m)
                                              / num_archives_to_process_margin_m) * args.initial_margin_m
            model.metric.m = effective_margin_m

        coeff = num_jobs
        if _iter < args.warmup_epochs > 0 and num_jobs > 1:
            coeff = float(_iter) * (num_jobs - 1) / args.warmup_epochs + 1.0

        for param_group in optimizer.param_groups:
            param_group['lr'] = effective_learning_rate * coeff

        if hvd.rank() == 0:
            lr = optimizer.param_groups[0]['lr']

            if args.metric == 'linear':
                logger.info(f'Iter: {_iter + 1}/{num_iters}  Epoch: {epoch:0.2f}/{args.num_epochs:0.1f}'
                            f'  ({percent:0.1f}% complete)  lr: {lr:0.5f}')
            else:
                logger.info(f'Iter: {_iter + 1}/{num_iters}  Epoch: {epoch:0.2f}/{args.num_epochs:0.1f}'
                            f'  ({percent:0.1f}% complete)  lr: {lr:0.5f} margin: {model.metric.m:0.4f}')

        train_dataset.set_iteration(_iter + 1)

        train_one_iteration(
            args=args,
            main_dir=args.dir,
            _iter=_iter + 1,
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            log_interval=500,
            len_train_sampler=len(train_sampler))

        if args.cleanup and hvd.rank() == 0:
            # do a clean up everything but the last 2 models, under certain conditions
            _remove_model(args.dir, _iter - 2, args.preserve_model_interval)

    if hvd.rank() == 0:
        ze_utils.force_symlink(f'raw_{num_iters}.pth', os.path.join(args.dir, 'models', 'model_final'))

    if args.cleanup and hvd.rank() == 0:
        logger.info(f'Cleaning up the experiment directory {args.dir}')
        for _iter in range(num_iters - 2):
            _remove_model(args.dir, _iter, args.preserve_model_interval)


if __name__ == '__main__':
    args = get_args()

    assert os.path.isdir(args.egs_dir), f'egs directory `{args.egs_dir}` does not exist.'

    try:
        train(args)
        ze_utils.wait_for_background_commands()
    except BaseException as e:
        # look for BaseException so we catch KeyboardInterrupt, which is
        # what we get when a background thread dies.
        if not isinstance(e, KeyboardInterrupt):
            traceback.print_exc()
        if os.path.exists('using_gpus.txt') and hvd.rank() == 0:
            logger.info('Removing using_gpus.txt file')
            os.remove('using_gpus.txt')
        sys.exit(1)

