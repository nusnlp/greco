import argparse
import math
import json

import numpy as np

from scipy.stats import t, pearsonr, spearmanr, kendalltau, rankdata, tmean, ttest_rel


def calculate_corr(values_a, values_b):
    if args.metric.lower().startswith('p'):
        result = pearsonr(values_a, values_b)[0]
    elif args.metric.lower().startswith('s'):
        result = spearmanr(values_a, values_b)[0]
    elif args.metric.lower().startswith('k'):
        result = kendalltau(values_a, values_b)[0]
    else:
        raise NotImplementedError("{} has not been implemented.".format(args.metric))

    return result

def williams_test(r12, r13, r23, n):
    """The Williams test (Evan J. Williams. 1959. Regression Analysis, volume 14. Wiley, New York, USA)
    
    A test of whether the population correlation r12 equals the population correlation r13.
    Significant: p < 0.05
    
    Arguments:
        r12 (float): correlation between x1, x2
        r13 (float): correlation between x1, x3
        r23 (float): correlation between x2, x3
        n (int): size of the population
        
    Returns:
        t (float): Williams test result
        p (float): p-value of t-dist
    """
    if r12 < r13:
        print('r12 should be larger than r13')
        return 1
    elif n <= 3:
        print('n should be larger than 3')
        return float('NaN')
    else:
        K = 1 - r12**2 - r13**2 - r23**2 + 2*r12*r13*r23
        denominator = np.sqrt(2*K*(n-1)/(n-3) + (((r12+r13)**2)/4)*((1-r23)**3))
        numerator = (r12-r13) * np.sqrt((n-1)*(1+r23))
        t_ = numerator / denominator
        p = 1 - t.cdf(t_, df=n-3)
        return p


def main(args):
    with open(args.system_A, encoding='utf-8') as f:
        system_A = json.load(f)
    with open(args.system_B, encoding='utf-8') as f:
        system_B = json.load(f)
    with open(args.target, encoding='utf-8') as f:
        target = json.load(f)
    assert len(system_A) == len(system_B) == len(target) 
    t_scores = []
    a_scores = []
    b_scores = []

    if args.metric.lower() in ['pearson', 'spearman']:
        key = 'score'
    elif args.metric.lower() in ['kendall']:
        key = 'rank'
    else:
        raise NotImplementedError("Correlation {} is not supported.".format(
                args.metric))
    print('Calculating correlation for {} sentences'.format(len(target)))
    for s_id, (sys_A, sys_B, target) in enumerate(zip(system_A, system_B, target)):
        score_t = []
        score_a = []
        score_b = []
        for sys_name, target_data in target.items():
            score_t.append(target_data[key])
            s_a = sys_A[sys_name][key]
            if math.isnan(s_a):
                s_a = 0
            score_a.append(s_a)
            s_b = sys_B[sys_name][key]
            if math.isnan(s_b):
                s_b = 0
            score_b.append(s_b)
        assert len(score_t) == len(score_a) == len(score_b)
        if args.sentence:
            t_scores.append(score_t)
            a_scores.append(score_a)
            b_scores.append(score_b)
        else:
            t_scores.extend(score_t)
            a_scores.extend(score_a)
            b_scores.extend(score_b)
   

    if args.sentence:
        a_corrs = []
        b_corrs = []
        w_p = []
        for score_a, score_b, score_t in zip(a_scores, b_scores, t_scores):
            corr_at = calculate_corr(score_a, score_t)
            corr_bt = calculate_corr(score_b, score_t)
            corr_ab = calculate_corr(score_a, score_b)
            if corr_at is None or corr_bt is None:
                print(score_a, score_b, score_t)
            if math.isnan(corr_at) and math.isnan(corr_bt):
                continue
            elif math.isnan(corr_at):
                print('Skipping due to sys A having NaN corr')
                print(score_a, score_t)
                continue
            elif math.isnan(corr_bt):
                print('Skipping due to sys B having NaN corr')
                print(score_b, score_t)
                continue
            w_p.append(williams_test(corr_at, corr_bt, corr_ab, len(score_t)))
            a_corrs.append(corr_at)
            b_corrs.append(corr_bt)
        print('System A corr:', tmean(a_corrs))
        print('System B corr:', tmean(b_corrs))
        print('======')
        print('T-test\t:', ttest_rel(a_corrs, b_corrs, alternative='greater')[1])
        print('Min W\t:', min(w_p))
        print('Mean W\t: ', tmean(w_p))
        print('Max W\t: ', max(w_p))
    else:
        corr_at = calculate_corr(a_scores, t_scores)
        corr_bt = calculate_corr(b_scores, t_scores)
        corr_ab = calculate_corr(a_scores, b_scores)
        print('System A corr:', tmean(corr_at))
        print('System B corr:', tmean(corr_bt))
        print("William's test: ", 
                williams_test(corr_at, corr_bt, corr_ab, len(t_scores)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--system_A', type=str, required=True, help='Rank JSON file for system A')
    parser.add_argument('--system_B', type=str, required=True, help='Rank JSON file for the BASELINE system')
    parser.add_argument('--target', type=str, required=True, help='Rank JSON file')
    parser.add_argument('--metric', type=str, default='spearman', help='Rank JSON file')
    parser.add_argument('--sentence', default=False, action='store_true', help='calculate correlation per source sentence')
    args = parser.parse_args()
    main(args)
