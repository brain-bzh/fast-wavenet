import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import manifold
from tqdm import tqdm
from scipy.spatial import distance
from librosa.core import resample

class Displayer :
    """
    A class that generates images to represent the data
    Please feed it one chunk per second, and call update() 24 times per second
    sr is the sample rate of the audio
    size is the amount of points in each image
    sound and data are the initial values
    """

    
    def __init__(self, sr, size, sound=None, data=None) :
        if not (type(sound) is np.ndarray and type(data) is np.ndarray) :
            sound, data = load()
            print("No data or no sound found, loading from database")
        sound, data = reject_outliers(sound, data)
        original_sound = sound
        self.sound = getSoundUndersampling(sound, sr) # Size is the number of points in every image
        self.data = data
        self.sp = manifold.SpectralEmbedding(3, affinity='precomputed')
        self.size = size
        self.sr = sr # This is the sample rate of the incoming audio
        self.data = np.append(data[:size], getUndersampling(self.data, self.sound, original_sound), axis=0)
        self.sound = np.append(sound[:size], self.sound)
        self.firstdata = data[:size]
        self.sim_mat, self.dist_mat, self.old_max = getBothMatrices(self.firstdata)
        self.endIndex = size
        self.initIndex = 0

    def display(self, arr=None) :
        """ 
        displays a 3d array made with a SpectralEmbedding
        if no array is provided, creates it 
        """
        if (arr == None) :
            arr = self.sp.fit_transform(self.sim_mat).T
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(arr[0], arr[1], arr[2], c=self.sound[self.initIndex:self.endIndex], cmap=plt.cm.spring) #cm.afmhot nice too
        plt.show()
        return

    def add_sample(self, s, d) :
        """
        adds a chunk of sound and data to the model
        """
        s, d = reject_outliers(s, d)
        self.sound = np.append(self.sound, getSoundUndersampling(s, self.sr))
        self.data = np.append(self.data, getUndersampling(d, self.sound, s), axis=0)
        return

    def update(self, new_sample=None) :
        """
        update the similarity matrix and the different values required to compute it
        you can pass it a sample but it is recomended to let it take the next one in the buffer
        """
        if (new_sample == None) :
            try :
                new_sample = self.data[self.endIndex]
                self.endIndex += 1
                self.initIndex += 1
            except :
                raise ValueError("Please add data before updating")
        sim_mat, dist_mat, old_max = self.sim_mat, self.dist_mat, self.old_max
        n = self.size
        new_sim = np.zeros((n,n))
        new_dist = np.zeros((n,n))
        dist = np.append(distance.cdist([new_sample], self.data[self.initIndex:self.endIndex-1]).reshape(n-1), [0])
        new_max = np.max(dist)
        if (new_max <= self.old_max) :
            new_sim[:-2,:-2] = sim_mat[:-2,:-2]
            d = np.ones(n) * old_max
            dist = d - dist
            new_sim[-1,:] = dist
            new_sim[:,-1] = dist
            new_dist[:-2,:-2] = dist_mat[:-2,:-2]
            new_dist[:,-1] = dist
            new_dist[-1,:] = dist
        else :
            new_dist[:-2,:-2] = dist_mat[:-2,:-2]
            new_dist[:,-1] = dist
            new_dist[-1,:] = dist
            d = np.ones(n) * new_max
            new_sim = d - new_dist
            self.old_max = new_max
        self.dist_mat = new_dist
        self.sim_mat = new_sim
        return

    def getSimMatrix(self) :
        return self.sim_mat

    def getArray(self) :
        return self.sp.fit_transform(self.sim_mat).T


def getBothMatrices(d) :
    """
    Creates the similarity matrix of data
    Used to update the matrix
    returns : a s*s matrix where s is the number of samples, representing how close the samples are
    """
    di = d
    mat = distance.cdist(d, d)
    m = np.max(mat)
    matrix = np.ones(mat.shape) * m
    return matrix-mat, mat, m

    
def load(i = 0) :
    """
    loads data located in current folder
    data is supposed to be created through make.py
    i is only here to allow for different files to be loaded
    returns : the sound array and the latent vector array
    """
    if (i == 0) :
        data = np.load("data.npz")
    else :
        data = np.load("data"+str(i)+".npz")
    data.allow_pickle = True
    return data["x"], data["y"]

def getSoundUndersampling(s, rate) :
    """
    Resamples the audio to match the image
    Required to properly resample the latent vecotr, and to compute its color
    """
    return resample(s, rate, 24*rate/len(s), res_type='fft')

    
def getUndersampling(data, sound, original_sound) :
    """
    Resamples the latent vector to 24 points
    """
    l = len(data)
    inters = [data[int(l/25 * k) : int(l/25 * (k+1))] for k in range(24)]
    weights = [original_sound[int(l/25*k) : int(l/25*(k+1))] for k in range(24)]
    for index, i in enumerate(weights) :
        i = np.ones(i.shape)*sound[index] - i
    l = []
    for i in range(len(inters)) :
        try :
            l.append(np.average(inters[i], axis=0, weights=weights[i]))
        except e:
            print(e)
    return l

def reject_outliers(s, d, m=1) :
    """
    supposedly removes non-meaningful points
    """
    i = [abs(s - np.mean(s)) < m * np.std(s)]
    return s[i], d[i]


def test() : #may be a few name typos, not entirely tested yet
    """ 
    example function
    """
    import make
    make.gen_save(5)
    s, d = load(5)
    disp = Displayer(11025, 1000, s, d)
    for i in range(2):
        make.gen_save(i) #this creates the data and saves it
    for i in range(2) :
        s, d = load(i) #loads the generated data
        disp.add_sample(s, d) #stores the data in the displayer

    disp.display() #displays the image on the screen
    for i in range(24*3) : # we have 6 seconds
        disp.update() # we push the next sample in
        if (i % 12 == 0) :
            disp.display() # we display the image from time to time (every 0.5sec)
    return



def generate_one_good_image() :
    import make
    make.gen_save(42)
    d = Displayer(11025, 1000)
    d.display()
