import numpy as np
from sklearn.metrics import mean_squared_error
import random
from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import pairwise_kernels
def maximum_mean_discrepancy(X, Y, kernel='rbf', gamma=0.01):
    K_XX = pairwise_kernels(X, metric=kernel, gamma=gamma)
    K_YY = pairwise_kernels(Y, metric=kernel, gamma=gamma)
    K_XY = pairwise_kernels(X, Y, metric=kernel, gamma=gamma)
    mmd = np.mean(K_XX) - 2 * np.mean(K_XY) + np.mean(K_YY)
    return mmd
class Detector():
    def __init__(self, window_size, args):
        self.window_size = window_size
        self.memory = {}  # store existing distributions
        self.memory_info = {} # store the distribution corresponding thresholds
        self.current_centroid = None
        self.N = []
        self.newsample = []
        self.args = args
        self.n_components = 1
    def addsample2memory(self, sample, seen):
        self.memory = {'sample': sample, 'centroid': np.mean(sample, axis=0)}
        self.current_centroid = self.memory['centroid']
        threshold = self.compute_threshold(sample, self.current_centroid, self.args.threshold + 1)
        self.memory_info = {'size': len(sample), 'threshold': threshold, 'seen': seen}
    def resample(self, new_sample):
        org = self.memory['sample']
        old = org
        seen = len(new_sample)
        if self.args.sample_method == 'random':
            if len(org) < self.args.memory_size:
                full = self.args.memory_size - len(org)
                org = np.vstack((org, new_sample[:full]))
                new_sample = new_sample[full:]
            if len(new_sample) != 0:
                candidates = np.vstack((org, new_sample))
                selected_indices = np.random.choice(candidates.shape[0], size=self.args.memory_size, replace=False)
                org = candidates[selected_indices]
        if self.args.sample_method == 'resorvior':
            if len(org) < self.args.memory_size:
                full = self.args.memory_size - len(org)
                org = np.vstack((org, new_sample[:full]))
                new_sample = new_sample[full:]
            if len(new_sample) != 0:
                for i in range(len(new_sample)):
                    jj = random.randint(0, self.memory_info['seen'] + i)
                    if jj < self.args.memory_size:
                        org[jj, :] = new_sample[i]
        if self.args.sample_method == 'distance':
            if len(org) < self.args.memory_size:
                full = self.args.memory_size - len(org)
                org = np.vstack((org, new_sample[:full]))
                new_sample = new_sample[full:]
            if len(new_sample) != 0:
                candidates = np.vstack((org, new_sample))
                dists = [maximum_mean_discrepancy(candidates[i].reshape(-1, 1), self.memory['centroid'].reshape(-1, 1)) for i in range(len(candidates))]
                indices = np.argsort(dists)[self.args.memory_size:][::-1]
                org = candidates[indices]

        self.memory['sample'] = org
        self.memory_info['threshold'] = self.compute_threshold(old, self.current_centroid, self.args.threshold)
        self.memory['centroid'] = np.mean(org, axis=0)
        self.current_centroid = self.memory['centroid']
        self.memory_info['seen'] += seen
    def updatememory(self):
        self.resample(self.newsample)
        self.newsample = []

    def compute_threshold(self, rep, centroid, threshold):
        MMD = [maximum_mean_discrepancy(rep[i].reshape(-1, 1), centroid.reshape(-1, 1)) for i in range(len(rep))]
        mse_quantile = np.quantile(MMD, self.args.quantile)
        threshold = threshold * mse_quantile
        return threshold
