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
num_blocks = 2
num_layers = 12
num_hidden = 256

duration = 2
if (duration*sample_rate/2) % 2 == 0 :
    num_time_samples = int(sample_rate*duration)
else :
    num_time_samples = int(sample_rate*duration) - 1
num_channels = 1
gpu_fraction = 1.0
num_classes = Quantification

model = Model(num_time_samples=num_time_samples,
              num_channels=num_channels,
              gpu_fraction=gpu_fraction,
              num_classes=num_classes,
              num_blocks=num_blocks,
              num_layers=num_layers,
              num_hidden=num_hidden)

model.restore()

print("start")

generator = Generator(model)

print("Done")

new_pred = generator.run([[np.random.randn()]], num_time_samples*2)



output.write_wav('1dur_generated.wav', new_pred[0], sample_rate)

#new_pred = generator.run([[np.random.randn()]], num_time_samples*5)
#output.write_wav('5dur_generated.wav', new_pred[0], sample_rate)

#new_pred = generator.run([[np.random.randn()]], num_time_samples*120)
#output.write_wav('120dur_generated.wav', new_pred[0], sample_rate)




def createData() :
    sound, latent = generator.getData([[np.random.randn()]], num_time_samples*2)
    save(sound, latent)
    

def save(sound, h) :
    x, y = sound, h
    np.savez("data.npz", x=x, y=y, allow_pickle=True)


def load() :
    data = np.load("data.npz")
    data.allow_pickle = True
    return data["x"], data["y"]
