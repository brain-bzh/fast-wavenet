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

generator = Generator(model)

#new_pred = generator.run([[np.random.randn()]], num_time_samples*2)

#output.write_wav('1dur_generated.wav', new_pred[0], sample_rate)


#print("generator ready")

#sound, latent = generator.getData([[np.random.randn()]], num_time_samples*2)

#print("Done")


def gen_save(i) :
    sound, latent = generator.getData([[np.random.randn()]], num_time_samples*2)
    save(sound, latent, i)


def save(sound, h, i=0) :
    x, y = sound, h
    np.savez("data"+str(i)+".npz", x=x, y=y, allow_pickle=True)


def load(i=0) :
    data = np.load("data"+str(i)+".npz")
    data.allow_pickle = True
    return data["x"], data["y"]
