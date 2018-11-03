#!/usr/bin/env python

import os
import librosa
import numpy as np
import cupy as cp
import chainer
import chainer.cuda
from chainer import Variable
import matplotlib.pyplot as plt
import parameters


def out_image(updater, enc, dec, rows, cols, seed, dst):
    @chainer.training.make_extension()
    def make_image(trainer):
        np.random.seed(seed)
        xp = enc.xp

        batch = updater.get_iterator('test').next()
        batchsize = len(batch)
        labels,inputs = xp.split(xp.asarray(batch),[1],axis=1)
        # 2次元目はinputとlabelを分けて保持していたが、分割操作により２次元目は不要になる。
        # Encoderのデータ入力形式（ndim=4）に合わせる必要あり
        # (samples,input_or_label, channel(スペクトログラムと位相）, 周波数, 時刻) から input_or_labelを除去する
        dataShape = list(labels.shape)
        dataShape.pop(1)
        labels = labels.reshape(dataShape)      #(1,1,2,1024,-1) -> (1,2,1024,-1)
        inputs = inputs.reshape(dataShape)
        # labels = xp.array([[[[]]]])
        # inputs = xp.array([])
        # for i in range(batchsize):
        #     #np.append(inputs,xp.asarray(batch[i][0]),axis=0)   #GPU計算の場合はarray操作はcupyで行う。cupyにはappendという関数はない
        #     xp.concatenate((inputs,xp.asarray(batch[i][0])),axis=0)
        #     xp.concatenate((labels,xp.asarray(batch[i][1])),axis=0)
        x_in = Variable(inputs)
        z = enc(x_in)
        x_out = dec(z)
        outputs = x_out.array
        if xp != np :
            #GPU計算してるなら計算後データをGPUメモリからCPU側メモリへ移動
            # cupyの対応していない関数などのせいでpyplotなどでエラーでるから
            inputs = cp.asnumpy(inputs)
            outputs = cp.asnumpy(outputs)
            labels= cp.asnumpy(labels)

        for i,input,output,label in zip(range(batchsize),inputs,outputs,labels):
            # 1figureに複数imageをプロット
            # 参考：https://stackoverflow.com/questions/46615554/how-to-display-multiple-images-in-one-figure-correctly
            rows = 3
            cols=2
            fig = plt.figure(figsize=(10,15))
            def subplot(imageData,title):
                subplot.counter +=1  #1~7
                s = list(imageData.shape)
                imageData  = imageData.reshape((s[-2],s[-1]))  # (ch,h,w) -> (h,w)
                ax = fig.add_subplot(rows,cols,subplot.counter)
                plt.imshow(imageData,cmap='gray')
                ax.set_title(title)
            subplot.counter = 0

            print('plotting preview images iter:{}'.format(trainer.updater.iteration))

            subplot(input[0],'input spectrogram')
            subplot(input[1],'input phase')
            subplot(output[0],'output spectrogram')
            subplot(output[1],'output phase')
            subplot(label[0],'labels spectrogram')
            subplot(label[1],'labels phase')

            preview_dir = '{}/preview'.format(dst)
            fileName = 'iter{:0>8}_{}'.format(trainer.updater.iteration,i)
            imagePath = os.path.join(preview_dir,fileName + '.jpg')
            if not os.path.exists(preview_dir):
                os.makedirs(preview_dir)
            plt.savefig(imagePath)

            D_input_abs = input[0]
            D_input_phase = input[1]
            D_input = D_input_abs * np.exp(1j*D_input_phase)    #xp.exp(1j*Dphase)
            y_input_hat = librosa.istft(D_input)
            inputWavPath = os.path.join(preview_dir,fileName + '_input.wav')
            librosa.output.write_wav(inputWavPath,y_input_hat,sr=parameters.sample_ratio)

            D_output_abs = output[0]
            D_output_phase = output[1]
            D_output = D_output_abs * np.exp(1j*D_output_phase)    #xp.exp(1j*Dphase)
            y_output_hat = librosa.istft(D_output)
            outputWavPath = os.path.join(preview_dir,fileName + '_output.wav')
            librosa.output.write_wav(outputWavPath,y_output_hat,sr=parameters.sample_ratio)

            if parameters.enable_output_labelWav:
                D_label_abs = label[0]
                D_label_phase = label[1]
                D_label = D_label_abs * np.exp(1j*D_label_phase)    #xp.exp(1j*Dphase)
                y_label_hat = librosa.istft(D_label)
                labelWavPath = os.path.join(preview_dir,fileName + '_label.wav')
                librosa.output.write_wav(labelWavPath,y_label_hat,sr=parameters.sample_ratio)


    return make_image
