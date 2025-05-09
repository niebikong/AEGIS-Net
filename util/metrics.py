from __future__ import print_function

import numpy as np


def cal_metric(known, novel, method=None):
    tp, fp, fpr_at_tpr95 = get_curve(known, novel, method)
    results = dict()
    mtypes = ['FPR', 'AUROC', 'DTERR', 'AUIN', 'AUOUT']

    results = dict()

    # FPR
    mtype = 'FPR'
    results[mtype] = fpr_at_tpr95

    # AUROC
    mtype = 'AUROC'
    tpr = np.concatenate([[1.], tp/tp[0], [0.]])
    fpr = np.concatenate([[1.], fp/fp[0], [0.]])
    results[mtype] = -np.trapz(1.-fpr, tpr)

    # DTERR
    mtype = 'DTERR'
    results[mtype] = ((tp[0] - tp + fp) / (tp[0] + fp[0])).min()

    # AUIN
    mtype = 'AUIN'
    denom = tp+fp
    denom[denom == 0.] = -1.
    pin_ind = np.concatenate([[True], denom > 0., [True]])
    pin = np.concatenate([[.5], tp/denom, [0.]])
    results[mtype] = -np.trapz(pin[pin_ind], tpr[pin_ind])

    # AUOUT
    mtype = 'AUOUT'
    denom = tp[0]-tp+fp[0]-fp
    denom[denom == 0.] = -1.
    pout_ind = np.concatenate([[True], denom > 0., [True]])
    pout = np.concatenate([[0.], (fp[0]-fp)/denom, [.5]])
    results[mtype] = np.trapz(pout[pout_ind], 1.-fpr[pout_ind])

    return results

def get_curve(known, novel, method=None):
    tp, fp = dict(), dict()
    fpr_at_tpr95 = dict()

    known.sort()
    novel.sort()

    end = np.max([np.max(known), np.max(novel)])
    start = np.min([np.min(known),np.min(novel)])

    all = np.concatenate((known, novel))
    all.sort()

    num_k = known.shape[0]
    num_n = novel.shape[0]

    if method == 'row':
        threshold = -0.5
    else:
        threshold = known[round(0.05 * num_k)]

    tp = -np.ones([num_k+num_n+1], dtype=int)
    fp = -np.ones([num_k+num_n+1], dtype=int)
    tp[0], fp[0] = num_k, num_n
    k, n = 0, 0
    for l in range(num_k+num_n):
        if k == num_k:
            tp[l+1:] = tp[l]
            fp[l+1:] = np.arange(fp[l]-1, -1, -1)
            break
        elif n == num_n:
            tp[l+1:] = np.arange(tp[l]-1, -1, -1)
            fp[l+1:] = fp[l]
            break
        else:
            if novel[n] < known[k]:
                n += 1
                tp[l+1] = tp[l]
                fp[l+1] = fp[l] - 1
            else:
                k += 1
                tp[l+1] = tp[l] - 1
                fp[l+1] = fp[l]

    j = num_k+num_n-1
    for l in range(num_k+num_n-1):
        if all[j] == all[j-1]:
            tp[j] = tp[j+1]
            fp[j] = fp[j+1]
        j -= 1

    fpr_at_tpr95 = np.sum(novel > threshold) / float(num_n)

    # 计算给定阈值下的准确率
    if threshold is not None:
        # 已知类别样本中被正确分类的数量
        tp_at_threshold = np.sum(known >= threshold)
        # 未知类别样本中被正确分类的数量
        tn_at_threshold = np.sum(novel < threshold)
        # 总样本数
        total_samples = num_k + num_n
        # 准确率
        accuracy = (tp_at_threshold + tn_at_threshold) / total_samples
    else:
        accuracy = None
    print('accuracy:', accuracy)
    
    return tp, fp, fpr_at_tpr95

def print_results(results, in_dataset, out_dataset, name, method):
    mtypes = ['FPR', 'DTERR', 'AUROC', 'AUIN', 'AUOUT']

    print('in_distribution: ' + in_dataset)
    print('out_distribution: '+ out_dataset)
    print('Model Name: ' + name)
    print('')

    print(' OOD detection method: ' + method)
    for mtype in mtypes:
        print(' {mtype:6s}'.format(mtype=mtype), end='')
    print('\n{val:6.2f}'.format(val=100.*results['FPR']), end='')
    print(' {val:6.2f}'.format(val=100.*results['DTERR']), end='')
    print(' {val:6.2f}'.format(val=100.*results['AUROC']), end='')
    print(' {val:6.2f}'.format(val=100.*results['AUIN']), end='')
    print(' {val:6.2f}\n'.format(val=100.*results['AUOUT']), end='')
    print('')

    # print(' {val:6.2f}'.format(val=100.*results['FPR']), end='')
    # print(' {val:6.2f}'.format(val=100.*results['DTERR']), end='')
    # print(' {val:6.2f}'.format(val=100.*results['AUROC']), end='')
    # print(' {val:6.2f}'.format(val=100.*results['AUIN']), end='')
    # print(' {val:6.2f}'.format(val=100.*results['AUOUT']), end='')

def print_all_results_tab(results, datasets, method):
    [print('{:6.2f}\t{:6.2f}\t{:6.2f}\t'.format(100.*result['FPR'], 100.*result['AUROC'], 100.*result['AUIN']), end='') for result in results]


def print_all_results(results, datasets, method):
    mtypes = ['FPR', 'AUROC', 'AUIN']#['FPR', 'DTERR', 'AUROC', 'AUIN', 'AUOUT']
    avg_results = compute_average_results(results)
    print(' OOD detection method: ' + method)
    print('             ', end='')
    for mtype in mtypes:
        print(' {mtype:6s}'.format(mtype=mtype), end='')
    for result, dataset in zip(results,datasets):
        print('\n{dataset:12s}'.format(dataset=dataset), end='')
        print(' {val:6.2f}'.format(val=100.*result['FPR']), end='')
        # print(' {val:6.2f}'.format(val=100.*result['DTERR']), end='')
        print(' {val:6.2f}'.format(val=100.*result['AUROC']), end='')
        print(' {val:6.2f}'.format(val=100.*result['AUIN']), end='')
        # print(' {val:6.2f}'.format(val=100.*result['AUOUT']), end='')

    print('\nAVG         ', end='')
    print(' {val:6.2f}'.format(val=100.*avg_results['FPR']), end='')
    # print(' {val:6.2f}'.format(val=100.*avg_results['DTERR']), end='')
    print(' {val:6.2f}'.format(val=100.*avg_results['AUROC']), end='')
    print(' {val:6.2f}'.format(val=100.*avg_results['AUIN']), end='')
    print()
    # [print('{:6.2f}\t{:6.2f}\t{:6.2f}\t'.format(100.*result['FPR'], 100.*result['AUROC'], 100.*result['AUIN']), end='') for result in results]
    # print()
    # print(' {val:6.2f}\n'.format(val=100.*avg_results['AUOUT']), end='')

def compute_average_results(all_results):
    mtypes = ['FPR', 'DTERR', 'AUROC', 'AUIN', 'AUOUT']
    avg_results = dict()

    for mtype in mtypes:
        avg_results[mtype] = 0.0

    for results in all_results:
        for mtype in mtypes:
            avg_results[mtype] += results[mtype]

    for mtype in mtypes:
        avg_results[mtype] /= float(len(all_results))

    return avg_results
