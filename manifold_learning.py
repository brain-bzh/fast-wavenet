from time import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
from sklearn import manifold
from sklearn.utils import check_random_state
from tqdm import tqdm
from scipy.spatial import distance


def load() :
    """
    loads data located in current folder
    data is supposed to be created through make.py
    returns : the sound array and the latent vector array
    """
    data = np.load("data.npz")
    data.allow_pickle = True
    return data["x"], data["y"]


# loads the sound and the data and undersamples it
sound, data = load()
data = data[:300]
sound = sound[:300]


def display() :
    """
    creates a SpectrogramEmbedding, the matrix and computes the
    3d figure to display and display it to the screen
    """
    m = manifold.SpectralEmbedding(3, affinity='precomputed')
    mat = getMatrix(data)
    d = m.fit_transform(mat).T
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(d[0], d[1], d[2], c=sound, cmap=plt.cm.spring) #cm.afmhot nice too
    plt.show()
    return 

def displayArr(d) :
    """ 
    displays a 3d array made with a SpectralEmbedding
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(d[0], d[1], d[2], c=sound, cmap=plt.cm.spring) #cm.afmhot nice too
    plt.show()
    return 

def getBothMatrices(d) :
    """
    Creates the similarity matrix of data
    Used to update the matrix
    returns : a s*s matrix where s is the number of samples, representing how close the samples are
    """
    mat = distance.cdist(d, d)
    m = np.max(mat)
    matrix = np.ones(mat.shape) * m
    return matrix-mat, mat, m

def getMatrix(d) :
    """
    Creates the similarity matrix of data
    returns : a s*s matrix where s is the number of samples, representing how close the samples are
    """
    mat = distance.cdist(d, d)
    m = np.max(mat)
    matrix = np.ones(mat.shape) * m
    return matrix-mat


def compare() :
    """
    Method comparing different methods of for SpectralEmbedding
    """
    tprint0, tprint1, tprint2, tprint3 = 0,0,0,0
    for i in tqdm(range(20)) :
        m0, m1 = manifold.SpectralEmbedding(3), manifold.SpectralEmbedding(3, affinity='precomputed')
        t0 = time()
        m0.fit_transform(data)
        t1 = time()
        tprint0 += t1-t0
        t0 = time()
        mat = getMatrix(data)
        t1 = time()
        m1.fit_transform(mat)
        t2 = time()
        tprint1 += t1-t0
        tprint2 += t2-t1
        tprint3 += t2-t0
    print("First method : " + str(tprint0/20))
    print("Second method :")
    print("\t mat : " + str(tprint1/20))
    print("\t fitting : " + str(tprint2/20))
    print("\t total : " + str(tprint3/20))
    return 



def getFitTime(l) :
    """
    Method getting the required time to fit a Spectral Embedding with a matrix of size l*l
    """
    r = 0
    if (l < 1) :
        return 0
    for i in range(5) :
        try :
            m = manifold.SpectralEmbedding(3, affinity='precomputed')
            _, d = load()
            mat = getMatrix(d[:int(l)])
            t0 = time()
            max=np.max(mat)
            m.fit_transform(mat)
            t1 = time()
            r += (t1-t0)
        except :
            return 0
    return (r/5)


def getUndersamplingRate(t, l,h, p=0.001) :
    c = getFitTime(int(h-l)/2)
    if (c==0) :
        print("No solution found")
        return
    print(str(c) + '\t' + str(h) + '\t' + str(l))
    if (abs(c - t) <= p) :
        return l
    if (c >= t) :
        return getUndersamplingRate(t, l, l+int(h-l)/2)
    if (c < t):
        return getUndersamplingRate(t, l+int(h-l)/2, h)
    


def updateMatrix(dist_mat, sim_mat, old_samples, new_sample, old_max) :
    n = dist_mat.shape[0]
    m = np.zeros((n,n))
    dist = np.append(distance.cdist([new_sample], old_samples[1:]).reshape(n-1), [0])
    new_max = np.max(dist)
    if (new_max <= old_max) :
        m[:-2,:-2] = sim_mat[:-2,:-2]
        d = np.ones(n) * old_max
        dist = d - dist
        m[-1,:] = dist
        m[:,-1] = dist
    else :
        m[:-2,:-2] = dist_mat[:-2,:-2]
        m[:,-1] = dist
        m[-1,:] = dist
        d = np.ones(n) * new_max
        m = d - m
    return m
