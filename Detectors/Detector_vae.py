import numpy as np
from sklearn.metrics import mean_squared_error
import random

class Detector():
    def __init__(self, window_size, feature_extracter, args):
        self.window_size = window_size
        self.memory = {}  # store existing distributions
        self.memory_info = {} # store the distribution corresponding thresholds
        self.current_centroid = None
        self.N = []
        self.newsample = []
        self.feature_extracter = feature_extracter
        self.args = args

    def addsample2memory(self, sample, rep, seen):
        self.memory = {'sample': sample, 'rep': rep, 'centroid': np.array([np.mean(rep)])}
        self.current_centroid = self.memory['centroid']
        threshold = self.compute_threshold(rep, self.current_centroid, self.args.threshold + 1)
        self.memory_info = {'size': len(sample), 'threshold': threshold, 'seen': seen}

    def resample(self, new_sample):
        seen = len(new_sample)
        new_sample = np.expand_dims(new_sample, axis=-1)
        org = self.memory['sample']
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
                dists = [np.sum((candidates[i] - self.memory['centroid']) ** 2) / candidates.shape[1] for i in range(len(candidates))]
                indices = np.argsort(dists)[self.args.memory_size:][::-1]
                org = candidates[indices]
        self.memory_info['threshold'] = self.compute_threshold(self.memory['rep'],self.current_centroid, self.args.threshold)
        self.memory['sample'] = org
        z_mean, z_log_sigma, z, pred = self.feature_extracter.predict(org)
        self.memory['rep'] = z_mean
        self.memory['centroid'] = np.array([np.mean(z_mean)])
        self.current_centroid = self.memory['centroid']
        self.memory_info['seen'] += seen

    def updatememory(self):
        self.resample(self.newsample)
        self.newsample = []

    def compute_threshold(self, rep, centroid, threshold):
        MSE = [np.sum((rep[i] - centroid) ** 2)/rep.shape[1] for i in range(len(rep))]
        mse_quantile = np.quantile(MSE, self.args.quantile)
        threshold = threshold * mse_quantile
        return threshold
