'''
Usage : train.py <sample_rate> <num_blocks> <num_layers> <num_hidden> <duration> <restore>

Parameters :
    - <sample_rate> : integer
    - <num_blocks> : integer
    - <num_layers> : integer
    - <num_hidden> : integer
    - <duration> : integer
    - <restore> : boolean, set to True if you want to train from a previous model
'''

from time import time
import datetime
import os
import numpy as np
import sys

from wavenet_vocoder import builder
from wavenet.utils import make_batch
from wavenet.models import Model, Generator

from librosa import output

checkpoint_dir = "/checkpoints"
server_dir = 'datasets/'
local_dir = 'datasets/'

sample_rate = int(sys.argv[1])
num_blocks = int(sys.argv[2])
num_layers = int(sys.argv[3])
num_hidden = int(sys.argv[4])
duration = int(sys.argv[5])
restore = bool(sys.argv[6])
Quantification = 256

if (duration*sample_rate/2) % 2 == 0 :
    num_time_samples = int(sample_rate*duration) - 1
else :
    num_time_samples = int(sample_rate*duration) - 1

def preprocess(path, first_seed, num_seed):
    WavListDir = [path]
    WavList = []
    for i in range(first_seed, first_seed + num_seed) :
        for t in WavListDir :
            WavList.append(t + str(i*2))
    return WavList

model = Model(num_time_samples=num_time_samples,
              num_channels=1,
              gpu_fraction=1.0,
              num_classes=Quantification,
              num_blocks=num_blocks,
              num_layers=num_layers,
              num_hidden=num_hidden)

if restore == True :
    print('Restoring model...')
    model.restore()

inputlist = []
targetlist = []
WavList = preprocess('Rhubarb/', 0, 10)

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

new_pred = generator.run([[np.random.randn()]], num_time_samples)
output.write_wav('{}sec_train.wav'.format(duration), new_pred[0], sample_rate)

#new_pred = generator.run([[np.random.randn()]], num_time_samples*5)
#output.write_wav('5_train.wav', new_pred[0], sample_rate)

#new_pred = generator.run([[np.random.randn()]], num_time_samples*120)
#output.write_wav('120_train.wav', new_pred[0], sample_rate)
