import glob
import argparse
import numpy as np
from Detectors.Detector_mean import Detector
import math
import utils.metrics as Evaluation_metrics
from btgym.btgym.research.model_based.model.rec import SSA
from utils.functions import preprocess, sliding_window
def main(args):
    error_margin = 864000
    no_CPs = 0
    no_preds = 0
    no_TPS = 0

    # Load and preprocess each data file
    for i in glob.glob(args.data):
        data = np.load(i, allow_pickle=True)
        _, train_dl, _, test_dl_2gal, cps = data['train_ts'], data['train_dl'], data['test_ts'], data['test_dl'], data[
            'label'].item()

        # Preprocess data for training
        train_dl_2gal = preprocess(train_dl[~np.isnan(train_dl).any(axis=1)], 1.5)

        # Preprocess data for testing
        test_dl_2gal = preprocess(test_dl_2gal[~np.isnan(test_dl_2gal).any(axis=1)], 1.5)
        ts, test_var_dl = test_dl_2gal[:, 0], test_dl_2gal[:, 1]

        # SSA initialisation for preprocessing module
        X = train_dl_2gal[:, 1]
        ssa = SSA(window=args.ssa_window, max_length=len(X))
        X_pred = ssa.reset(X)
        X_pred = ssa.transform(X_pred, state=ssa.get_state())
        reconstructeds = X_pred.sum(axis=0)
        residuals = X - reconstructeds
        resmean = residuals.mean()
        M2 = ((residuals - resmean) ** 2).sum()

        # Initialisation for feature extraction module
        reconstructeds = sliding_window(X, args.ws, args.step)
        memory = reconstructeds
        if len(reconstructeds) > args.memory_size:
            random_indices = np.random.choice(len(reconstructeds), size=args.memory_size, replace=False)
            memory = memory[random_indices]
        detector = Detector(args.ws, args)
        z_mean = np.mean(memory, axis=1).reshape(-1,1)
        detector.addsample2memory(memory, z_mean, len(memory))

        ctr = 0
        step = args.bs
        scores = [0]*len(ts)
        outliers = []
        filtered = []
        sample = np.empty((0, args.ws))
        collection_period = 10**18
        detected = False
        thresholds = [0]*len(ts)

        # Ground truth window with error margin
        gt_margin = []
        for tt in cps:
            closest_element = ts[ts < tt].max()
            idx = np.where(ts == closest_element)[0][0]
            gt_margin.append((ts[idx - 10], tt + error_margin, tt))

        # Main detection loop
        while ctr < test_var_dl.shape[0]:
            new = test_var_dl[ctr:ctr + step]
            updates = ssa.update(new)
            updates = ssa.transform(updates, state=ssa.get_state())[:, args.ssa_window-1:]
            reconstructed = updates.sum(axis=0)
            residual = new - reconstructed
            residuals = np.concatenate([residuals, residual])

            # Update outlier thresholds dynamically
            for i1 in range(len(new)):
                delta = residual[i1] - resmean
                resmean += delta / (ctr + i1 + len(train_dl_2gal))
                M2 += delta * (residual[i1] - resmean)
                stdev = math.sqrt(M2 / (ctr + i1 + len(train_dl_2gal) - 1))
                threshold_upper = resmean + args.out_threshold * stdev
                threshold_lower = resmean - args.out_threshold * stdev
                # Outlier filtering
                if residual[i1] > threshold_upper or residual[i1] < threshold_lower:
                    outliers.append(ctr + i1)
                    filtered.append(np.mean(filtered[-10:] if len(filtered)>10 else 0))
                else:
                    filtered.append(new[i1])

            # Detection phase
            if collection_period > args.min_requirement:
                if ctr == 0:
                    window = np.array(filtered)
                else:
                    window = np.array(filtered[-args.ws - step + 1:])
                if len(window) <= args.ws:
                    break
                window = sliding_window(window, args.ws, args.step)
                z_mean = np.mean(window, axis=1).reshape(-1,1)

                for aa in range(len(z_mean)):
                    score = ((z_mean[aa]-detector.current_centroid)**2)[0]
                    scores[ctr+aa*args.step:ctr+aa*args.step+args.step] = [score]*args.step
                    thresholds[ctr+aa*args.step:ctr+aa*args.step+args.step] = [detector.memory_info['threshold']]*args.step
                    if score > detector.memory_info['threshold']:
                        detector.N.append(ctr + aa*args.step)
                        collection_period = 0
                        detected = True
                        filtered = filtered[:-len(z_mean) + aa*args.step + 1]
                        detector.newsample = []
                        break
                    else:
                        detector.newsample.append(window[aa])

                # update the rep and threshold for the current distribution
                if collection_period > args.min_requirement:
                    detector.updatememory()

            # Collection phase
            elif collection_period < args.min_requirement:
                if len(sample) == 0:
                    raw = np.array(filtered[-step + 1:])
                else:
                    raw = np.array(filtered[-args.ws - step + 1:])
                if len(raw) <= args.ws:
                    break
                window = sliding_window(raw, args.ws, args.step)
                if collection_period + step < args.min_requirement:
                    sample = np.concatenate((sample, window))
                    collection_period += step
                else:
                    sample = np.concatenate((sample, window))
                    if len(sample) > args.memory_size:
                        random_indices = np.random.choice(len(sample), size=args.memory_size, replace=False)
                        sample = sample[random_indices]
                    z_mean = np.mean(sample, axis=1).reshape(-1,1)
                    detector.addsample2memory(sample, z_mean, len(sample))
                    collection_period = 10**18
                    sample = np.empty((0, args.ws))

            if detected:
                ctr += aa * args.step + 1
                detected = False
            elif len(test_var_dl) - ctr <= args.bs:
                break
            elif len(test_var_dl) - ctr <= 2 * args.bs:
                ctr += args.bs
                step = len(test_var_dl) - ctr
            else:
                ctr += args.bs

        # Evaluate each prediction against ground truth window
        preds = detector.N
        no_CPs += len(cps)
        no_preds += len(preds)
        mark = []
        for j in preds:
            timestamp = ts[j]
            for l in gt_margin:
                if timestamp >= l[0] and timestamp <= l[1]:
                    if l not in mark:
                        mark.append(l)
                    else:
                        no_preds -= 1
                        continue
                    no_TPS += 1

    rec = Evaluation_metrics.recall(no_TPS, no_CPs)
    FAR = Evaluation_metrics.False_Alarm_Rate(no_preds, no_TPS)
    prec = Evaluation_metrics.precision(no_TPS, no_preds)
    f2score = Evaluation_metrics.F2_score(rec, prec)
    np.savez(args.outfile, rec=rec, FAR=FAR, prec=prec, f2score=f2score)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MOCPD-Mean evaluation on 0.2 gph data')
    parser.add_argument('--data', type=str, default='./data/*.npz', help='directory of data')
    parser.add_argument('--sample_method', type=str, default='random', help='sampling method')
    parser.add_argument('--ssa_window', type=int, default=5, help='n_components for ssa preprocessing')
    parser.add_argument('--bs', type=int, default=150, help='buffer size for ssa')
    parser.add_argument('--ws', type=int, default=100, help='window size')
    parser.add_argument('--step', type=int, default=10, help='step')
    parser.add_argument('--min_requirement', type=int, default=500, help='window size')
    parser.add_argument('--memory_size', type=int, default=75, help='memory size per distribution ')
    parser.add_argument('--out_threshold', type=float, default=2, help='threshold for outlier filtering')
    parser.add_argument('--threshold', type=float, default=4, help='threshold')
    parser.add_argument('--quantile', type=float, default=0.975, help='quantile')
    parser.add_argument('--fixed_outlier', type=float, default=1, help='preprocess outlier filter')
    parser.add_argument('--outfile', type=str, default='mean_02_100', help='name of file to save results')
    args = parser.parse_args()

    main(args)