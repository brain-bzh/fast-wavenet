'''
Usage : generating.py <sample_rate> <num_blocks> <num_layers> <num_hidden> <duration>
'''

from time import time
import datetime
import os
import numpy as np
import sys

from wavenet.utils import make_batch
from wavenet.models import Model, Generator
import tensorflow as tf

from librosa import output

sample_rate = int(sys.argv[1])
num_blocks = int(sys.argv[2])
num_layers = int(sys.argv[3])
num_hidden = int(sys.argv[4])
duration = int(sys.argv[5])

Quantification = 256
if (duration*sample_rate/2) % 2 == 0 :
    num_time_samples = int(sample_rate*duration)
else :
    num_time_samples = int(sample_rate*duration) - 1

#Initialize the model before restoring it
model = Model(num_time_samples=num_time_samples,
              num_channels=1,
              gpu_fraction=1.0,
              num_classes=Quantification,
              num_blocks=num_blocks,
              num_layers=num_layers,
              num_hidden=num_hidden)

#Restoring the model
model.restore()

#Creating the Generator to make the prediction
generator = Generator(model)

seed = np.random.uniform(low=-1.0, high=1.0)

random_input = [[seed]]
random_inputs = []
for i in range(num_hidden-1):
    random_inputs.append(0)
random_inputs.append(seed)
random_inputs = np.asarray(random_inputs).reshape(num_hidden,1).tolist()

print('Making first prediction...')
tic = time()
combined_pred = generator.run(random_input, sample_rate).tolist()

for t in range(duration - 1) :
    print('Making next prediction : {}/{}... at time : {} seconds'.format(t+2, duration, time() - tic))
    if t % 5 == 0 :
        curent_input = [[np.random.uniform(low=-1.0, high=1.0)]]
    else :
        curent_input = [[combined_pred[-1]]]
    #curent_inputs = np.asarray(combined_pred[len(combined_pred)-num_hidden:]).reshape(num_hidden,1).tolist()
    curent_pred = generator.run(curent_input, sample_rate).tolist()
    combined_pred.extend(curent_pred)

output.write_wav('{}s_generated.wav'.format(duration), np.asarray(combined_pred), sample_rate)
print('Total time of generation is {}'.format(time()-tic))
print('Wav file created at : ' + '{}s_generated.wav'.format(duration))

#new_pred = generator.run([[np.random.randn()]], num_time_samples*5)
#output.write_wav('5_generated.wav', new_pred[0], sample_rate)

#new_pred = generator.run([[np.random.randn()]], num_time_samples*120)
#output.write_wav('120_generated.wav', new_pred[0], sample_rate)
