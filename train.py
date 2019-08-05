from time import time
import datetime
import os
import numpy as np


from wavenet.utils import make_batch
from wavenet.models import Model, Generator
import tensorflow as tf

from librosa import output

checkpoint_dir = "/checkpoints"
server_dir = 'datasets/'
local_dir = 'datasets/'

sample_rate = 11025

Quantification = 256
WavListDir = ['August/']
WavList = []
for i in range(2,3) :
    for t in WavListDir :
        WavList.append(t + str(i))
#WavList = ['ambient','synth2','Floating_Points', 'AphexTwinDrum', '909ishShortBeautyKick', 'BassRave']

duration = 2
if (duration*sample_rate/2) % 2 == 0 :
    num_time_samples = int(sample_rate*duration)
else :
    num_time_samples = int(sample_rate*duration) - 1
num_channels = 1
gpu_fraction = 1.0
num_classes = Quantification
num_blocks = 2
num_layers = 12
num_hidden = 256

model = Model(num_time_samples=num_time_samples,
              num_channels=num_channels,
              gpu_fraction=gpu_fraction,
              num_classes=num_classes,
              num_blocks=num_blocks,
              num_layers=num_layers,
              num_hidden=num_hidden)

inputlist = []
targetlist = []

for w in WavList :

    path = 'assets/' + w +'.wav'
    inputs, targets = make_batch(path, sample_rate, duration=duration)
    inputlist.append(inputs)
    targetlist.append(targets)

inputlist = np.stack(inputlist)
targetlist = np.stack(targetlist)

print(inputlist.shape,targetlist.shape)

train_step, losses = model.train(inputlist.reshape((inputlist.shape[0],inputlist.shape[2],1)), targetlist.reshape((targetlist.shape[0],-1)))

generator = Generator(model)

new_pred = generator.run([[np.random.randn()]], num_time_samples*2)
output.write_wav('2dur_train.wav', new_pred[0], sample_rate)

#new_pred = generator.run([[np.random.randn()]], num_time_samples*5)
#output.write_wav('5dur_train.wav', new_pred[0], sample_rate)

#new_pred = generator.run([[np.random.randn()]], num_time_samples*120)
#output.write_wav('120dur_train.wav', new_pred[0], sample_rate)
