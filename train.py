#!/usr/bin/env python

# python train.py -g 0 -i ./facade/base --out result_facade --snapshot_interval 10000

from __future__ import print_function
import argparse
from datetime import datetime
import os
import chainer
from chainer import training
from chainer.training import extensions

import util
from net import Discriminator
from net import Encoder
from net import Decoder
from updater import VoiceP2PUpdater

from data_loader import Vp2pDataset
from image_visualizer import out_image

def main():
    parser = argparse.ArgumentParser(description='chainer implementation of pix2pix')
    parser.add_argument('--batchsize', '-b', type=int, default=1,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=200,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--dataset', '-i', default='./facade/base',
                        help='Directory of image files.')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--snapshot_interval', type=int, default=1000,
                        help='Interval of snapshot')
    parser.add_argument('--display_interval', type=int, default=100,
                        help='Interval of displaying log to console')
    parser.add_argument('--n_processes', type=int, default=None,
                        help='processes of chainer.iterators.MultiprocessIterator')
    parser.add_argument('--shared_mem', type=int, default=None,
                        help='shared memory per data, for chainer.iterators.MultiprocessIterator. None means auto ajust.')
    parser.add_argument('--audio_dataset_second', type=int, default=None,
                        help='time length(second) of train audio data .')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    args.out = os.path.join(args.out,datetime.now().strftime("%Y%m%d_%H%M%S"))
    util.audio_dataset_second  =  args.audio_dataset_second
    if args.batchsize > 1 :  assert util.audio_dataset_second != None , "when minibatch training (e.g. --batchsize > 1), --audio_dataset_second option is required."

    # Set up a neural network to train
    enc = Encoder(in_ch=2)
    dec = Decoder(out_ch=2)
    dis = Discriminator(in_ch=2, out_ch=2)
    
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        enc.to_gpu()  # Copy the model to the GPU
        dec.to_gpu()
        dis.to_gpu()

    # Setup an optimizer
    def make_optimizer(model, alpha=0.0002, beta1=0.5):
        optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1)
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.WeightDecay(0.00001), 'hook_dec')
        return optimizer
    opt_enc = make_optimizer(enc)
    opt_dec = make_optimizer(dec)
    opt_dis = make_optimizer(dis)

    train_d = Vp2pDataset(args.dataset + "/train")
    test_d = Vp2pDataset(args.dataset + "/test")
    train_iter = chainer.iterators.MultiprocessIterator(train_d, args.batchsize, n_processes=args.n_processes,shared_mem=args.shared_mem)
    test_iter = chainer.iterators.MultiprocessIterator(test_d, args.batchsize, n_processes=args.n_processes,shared_mem=args.shared_mem)
    # train_iter = chainer.iterators.MultiprocessIterator(train_d, args.batchsize)
    # test_iter = chainer.iterators.MultiprocessIterator(test_d, args.batchsize)

    # Set up a trainer
    updater = VoiceP2PUpdater(
        models=(enc, dec, dis),
        iterator={
            'main': train_iter,
            'test': test_iter},
        optimizer={
            'enc': opt_enc, 'dec': opt_dec, 
            'dis': opt_dis},
        device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    snapshot_interval = (args.snapshot_interval, 'iteration')
    display_interval = (args.display_interval, 'iteration')
    trainer.extend(extensions.snapshot(
        filename='snapshot_iter_{.updater.iteration}.npz'),
                   trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        enc, 'enc_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        dec, 'dec_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        dis, 'dis_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.LogReport(trigger=display_interval))
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'enc/loss', 'dec/loss', 'dis/loss',
    ]), trigger=display_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(
        out_image(
            updater, enc, dec,
            5, 5, args.seed, args.out),
        trigger=snapshot_interval)

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()

if __name__ == '__main__':
    main()
