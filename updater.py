#!/usr/bin/env python

from __future__ import print_function

import chainer
import chainer.functions as F
from chainer import Variable

import numpy as np
from PIL import Image

from chainer import cuda
from chainer import function
from chainer.utils import type_check
import numpy

class VoiceP2PUpdater(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.enc, self.dec, self.dis = kwargs.pop('models')
        super(VoiceP2PUpdater, self).__init__(*args, **kwargs)


    def loss_enc(self, enc, x_out, t_out, y_out, lam1=100, lam2=1):
        batchsize,_,w,h = y_out.data.shape
        loss_rec = lam1*(F.mean_absolute_error(x_out, t_out))
        loss_adv = lam2*F.sum(F.softplus(-y_out)) / batchsize / w / h
        loss = loss_rec + loss_adv
        chainer.report({'loss': loss}, enc)
        return loss
        
    def loss_dec(self, dec, x_out, t_out, y_out, lam1=100, lam2=1):
        batchsize,_,w,h = y_out.data.shape
        loss_rec = lam1*(F.mean_absolute_error(x_out, t_out))
        loss_adv = lam2*F.sum(F.softplus(-y_out)) / batchsize / w / h
        loss = loss_rec + loss_adv
        chainer.report({'loss': loss}, dec)
        return loss
        
        
    def loss_dis(self, dis, y_in, y_out):
        batchsize,_,w,h = y_in.data.shape
        
        L1 = F.sum(F.softplus(-y_in)) / batchsize / w / h
        L2 = F.sum(F.softplus(y_out)) / batchsize / w / h
        loss = L1 + L2
        chainer.report({'loss': loss}, dis)
        return loss

    def update_core(self):        
        enc_optimizer = self.get_optimizer('enc')
        dec_optimizer = self.get_optimizer('dec')
        dis_optimizer = self.get_optimizer('dis')
        
        enc, dec, dis = self.enc, self.dec, self.dis
        xp = enc.xp

        batch = self.get_iterator('main').next()
        batchsize = len(batch)
        in_ch = batch[0][0].shape[0]
        out_ch = batch[0][1].shape[0]
        # w_in = 256
        # w_out = 256
        w_in = batch[0][0].shape[1]
        h_in = batch[0][0].shape[2]
        w_out = w_in
        h_out = h_in

        x_in = xp.zeros((batchsize, in_ch, w_in, h_in)).astype("f")
        t_out = xp.zeros((batchsize, out_ch, w_out, h_out)).astype("f")
        
        for i in range(batchsize):
            #todo batchsize ２以上だとスペクトログラム画像の横サイズ（時間方向）がサンプル間で異なることがあり、ndarrayへマージできず学習がエラー停止する。強制的にデータ除外することで対応中。
            if(x_in[i,:].shape != batch[i][0].shape) :
                print("skipped training_audio_sample because spectrogram shape does not match.(comes from program bug.)\r\n expected:{}, actual:{}"
                      .format(x_in[i,:].shape ,batch[i][0].shape))
                continue

            x_in[i,:] = xp.asarray(batch[i][0])
            t_out[i,:] = xp.asarray(batch[i][1])
        x_in = Variable(x_in)
        
        z = enc(x_in)
        x_out = dec(z)

        y_fake = dis(x_in, x_out)
        y_real = dis(x_in, t_out)


        enc_optimizer.update(self.loss_enc, enc, x_out, t_out, y_fake)
        for z_ in z:
            z_.unchain_backward()
        dec_optimizer.update(self.loss_dec, dec, x_out, t_out, y_fake)
        x_in.unchain_backward()
        x_out.unchain_backward()
        dis_optimizer.update(self.loss_dis, dis, y_real, y_fake)
