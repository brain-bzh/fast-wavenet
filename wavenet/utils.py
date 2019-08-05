import numpy as np
from librosa import load



def normalize(data):
    temp = np.float32(data) - np.min(data)
    out = (temp / np.max(temp) - 0.5) * 2
    return out


def make_batch(path, sample_rate, duration):

    # Sample rate to test with 8000,16000,22050,44100

    quantification = 256

    data = load(path, sample_rate, duration = duration)[0]

    #silence = np.asarray([0]*int((sample_rate/2)))

    #silenced_data = np.append(silence, data)

    #print('The number of sample is ' + str(data.shape))

    #print('The sample rate is ' + str(sample_rate))

    data_ = normalize(data)
    # data_f = np.sign(data_) * (np.log(1 + 255*np.abs(data_)) / np.log(1 + 255))

    bins = np.linspace(-1, 1, quantification)
    # Quantize inputs.
    inputs = np.digitize(data_[0:-1], bins, right=False) - 1
    inputs = bins[inputs][None, :, None]

    # Encode targets as ints.
    targets = (np.digitize(data_[1::], bins, right=False) - 1)[None, :]

    return inputs, targets
