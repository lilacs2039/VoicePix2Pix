#!/usr/bin/env python

from __future__ import print_function

import chainer
import chainer.functions as F
import chainer.links as L

# U-net https://arxiv.org/pdf/1611.07004v1.pdf

# convolution-batchnormalization-(dropout)-relu
from util import DOF


class CBR(chainer.Chain):
    def __init__(self, ch0, ch1, bn=True, sample='down', activation=F.relu, dropout=False):
        self.bn = bn
        self.activation = activation
        self.dropout = dropout
        layers = {}
        w = chainer.initializers.Normal(0.02)
        if sample=='down':
            layers['c'] = L.Convolution2D(ch0, ch1, 4, 2, 1, initialW=w)
        else:
            layers['c'] = L.Deconvolution2D(ch0, ch1, 4, 2, 1, initialW=w)
        if bn:
            layers['batchnorm'] = L.BatchNormalization(ch1)
        super(CBR, self).__init__(**layers)
        
    def __call__(self, x):
        h = self.c(x)
        if self.bn:
            h = self.batchnorm(h)
        if self.dropout:
            h = F.dropout(h)
        if not self.activation is None:
            h = self.activation(h)
        return h


class Encoder(chainer.Chain):
    def __init__(self, in_ch):
        layers = {}
        w = chainer.initializers.Normal(0.02)
        layers['c0'] = L.Convolution2D(in_ch, DOF[0], 3, 1, 1, initialW=w)
        layers['c1'] = CBR(DOF[0], DOF[1], bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c2'] = CBR(DOF[1], DOF[2], bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c3'] = CBR(DOF[2], DOF[3], bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c4'] = CBR(DOF[3], DOF[4], bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c5'] = CBR(DOF[4], DOF[5], bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c6'] = CBR(DOF[5], DOF[6], bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c7'] = CBR(DOF[6], DOF[7], bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        super(Encoder, self).__init__(**layers)

    def __call__(self, x):
        hs = [F.leaky_relu(self.c0(x))]
        for i in range(1,8):
            hs.append(self['c%d'%i](hs[i-1]))
        return hs

class Decoder(chainer.Chain):
    def __init__(self, out_ch):
        layers = {}
        w = chainer.initializers.Normal(0.02)
        layers['c0'] = CBR(DOF[7], DOF[6], bn=True, sample='up', activation=F.relu, dropout=True)
        layers['c1'] = CBR(DOF[6] * 2, DOF[5], bn=True, sample='up', activation=F.relu, dropout=True)
        layers['c2'] = CBR(DOF[5] * 2, DOF[4], bn=True, sample='up', activation=F.relu, dropout=True)
        layers['c3'] = CBR(DOF[4] * 2, DOF[3], bn=True, sample='up', activation=F.relu, dropout=False)
        layers['c4'] = CBR(DOF[3] * 2, DOF[2], bn=True, sample='up', activation=F.relu, dropout=False)
        layers['c5'] = CBR(DOF[2] * 2, DOF[1], bn=True, sample='up', activation=F.relu, dropout=False)
        layers['c6'] = CBR(DOF[1] * 2, DOF[0], bn=True, sample='up', activation=F.relu, dropout=False)
        layers['c7'] = L.Convolution2D(DOF[0] * 2, out_ch, 3, 1, 1, initialW=w)
        super(Decoder, self).__init__(**layers)

    def __call__(self, hs):
        h = self.c0(hs[-1])
        for i in range(1,8):
            h = F.concat([h, hs[-i-1]])
            if i<7:
                h = self['c%d'%i](h)
            else:
                h = self.c7(h)
        return h



class Discriminator(chainer.Chain):
    def __init__(self, in_ch, out_ch):
        layers = {}
        w = chainer.initializers.Normal(0.02)
        layers['c0_0'] = CBR(in_ch, 32, bn=False, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c0_1'] = CBR(out_ch, 32, bn=False, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c1'] = CBR(64, 128, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c2'] = CBR(128, 256, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c3'] = CBR(256, 512, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c4'] = L.Convolution2D(512, 1, 3, 1, 1, initialW=w)
        super(Discriminator, self).__init__(**layers)

    def __call__(self, x_0, x_1):
        h = F.concat([self.c0_0(x_0), self.c0_1(x_1)])
        h = self.c1(h)
        h = self.c2(h)
        h = self.c3(h)
        h = self.c4(h)
        #h = F.average_pooling_2d(h, h.data.shape[2], 1, 0)
        return h
