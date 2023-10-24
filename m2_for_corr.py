import ast
import argparse
import json
import math
import os
import statistics
from operator import itemgetter

from scipy.stats import rankdata


def parse_errant_report(filename):
    print('parsing {}...'.format(filename))
    docs = []
    with open(filename, encoding='utf-8') as f:
        cur_id = -1
        doc = {}
        for line in f:
            if line.startswith('SENTENCE'):
                cur_id = int(line.split()[1])
            elif line.startswith('^^'):
                assert len(docs) == cur_id
                doc['id'] = cur_id
                doc['ann_id'] = int(line.split('REF')[1].strip().split()[0])
                docs.append(doc)
                doc = {}
            elif line.startswith('-------'):
                continue
            elif line.startswith('\n'):
                break
            else:
                val = ':'.join(line.split(':')[1:])
                if line.startswith('HYPOTHESIS'):
                    doc['hyp'] = ast.literal_eval(val.strip())
                elif line.startswith('REFERENCE'):
                    doc['ref'] = ast.literal_eval(val.strip())
                elif line.startswith('Local TP/FP/FN'):
                    doc['loc_count'] = val.strip().split()
                elif line.startswith('Local P/R/F0.5'):
                    doc['loc_f'] = val.strip().split()
                elif line.startswith('Global TP/FP/FN'):
                    doc['glob_count'] = val.strip().split()
                elif line.startswith('Global P/R/F0.5'):
                    doc['glob_f'] = val.strip().split()
                else:
                    raise NotImplementedError(
                            "Line {} is not handled".format(line[:15])
                        )

    return docs


def parse_m2scorer_report(filename):
    print('parsing {}...'.format(filename))
    docs = []
    with open(filename, encoding='utf-8') as f:
        cur_id = 0
        doc = {}
        dic_keys = ['fHalf', 'prec', 'rec', 'hyp', 'ref', 'src', 'text']
        starting_tmp = {
            'fHalf': [],
            'prec': [],
            'rec': [],
            'cor_hyp': [],
            'hyp': [],
            'ref': [],
            'src': [],
            'text': []
        }
        tmp = {k: v.copy() for k, v in starting_tmp.items()}
        for line in f:
            line = line.strip()
            if line.startswith('>> Chosen Annotator'):
                def calculate_fHalf(tp, p, gold):
                    precision = 1 if p == 0 else float(tp) / p
                    recall = 1 if gold == 0 else float(tp) / gold
                    f_half = 0 if precision + recall == 0 else (1 + 0.5 * 0.5) * precision * recall / (0.5 * 0.5 * precision + recall)
                    return precision, recall, f_half

                doc['id'] = cur_id
                doc['ann_id'] = int(line.split(':')[-1].strip())
                tp = [len(c) for c in tmp['cor_hyp']]
                p = [len(h) for h in tmp['hyp']]
                true_edits = [len(g) for g in tmp['ref']]
                # doc['loc_count'] = [tp, p - tp, true_edits - tp]
                all_loc_scores = [calculate_fHalf(*s) for s in zip(tp, p, true_edits)]
                ann_id, score = max(enumerate(all_loc_scores), key=itemgetter(1))
                doc['loc_f'] = score
                for k in dic_keys[3:]:
                    doc[k] = tmp[k][ann_id]
                
                ann_id, fHalf = max(enumerate(tmp['fHalf']), key=itemgetter(1))
                doc['glob_f'] = [tmp['prec'][ann_id], tmp['rec'][ann_id], fHalf]
                
                tmp = {k: v.copy() for k, v in starting_tmp.items()}
                docs.append(doc)
                doc = {}
                cur_id += 1
            elif ':' in line:
                val = ':'.join(line.split(':')[1:])
                annot_id = -1
                if line.startswith('>> Annotator'):
                    annot_id = int(val)
                elif line.startswith('f_0.5'):
                    fHalf = float(val.strip())
                    tmp['fHalf'].append(fHalf)
                elif line.startswith('precision'):
                    prec = float(val.strip())
                    tmp['prec'].append(prec)
                elif line.startswith('recall'):
                    rec = float(val.strip())
                    tmp['rec'].append(rec)
                elif line.startswith('EDIT SEQ'):
                    tmp['hyp'].append(ast.literal_eval(val.strip()))
                elif line.startswith('CORRECT EDITS'):
                    tmp['cor_hyp'].append(ast.literal_eval(val.strip()))
                elif line.startswith('GOLD EDITS'):
                    tmp['ref'].append(ast.literal_eval(val.strip()))
                elif line.startswith('SOURCE'):
                    tmp['src'].append(val.strip())
                elif line.startswith('HYPOTHESIS'):
                    tmp['text'].append(val.strip())
                # loc_count is not implemented yet as it's not used
    return docs


def main(args):
    if args.scorer == 'errant':
        parser = parse_errant_report
    elif args.scorer == 'm2scorer':
        parser = parse_m2scorer_report
    else:
        raise NotImplementedError("Unknown {} scorer"\
            .format(args.scorer))
    
    first = True
    for filepath in os.listdir(args.data_dir):
        system_name, _ = os.path.splitext(
                            os.path.basename(filepath))
        docs = parser(os.path.join(args.data_dir, filepath))
        if first:
            all_data = [{} for _ in docs]
        for s_id, doc in enumerate(docs):
            all_data[s_id][system_name] = {
                'score': doc['loc_f'][-1]
            }
        first = False
    
    for s_dict in all_data:
        scores = [s['score'] for s in s_dict.values()]
        sys_names = s_dict.keys()
        asc_ranks = rankdata(scores, method='min')
        max_rank = max(asc_ranks)
        dsc_ranks = [int(max_rank - s + 1) for s in asc_ranks]
        for sy_id, sys_info in enumerate(s_dict.values()):
            sys_info['rank'] = dsc_ranks[sy_id]
    
    with open(args.output_path, 'w', encoding='utf-8') as out:
        json.dump(all_data, out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='XML file')
    parser.add_argument('--output_path', type=str, help='Output path')
    parser.add_argument('--scorer', type=str, choices=['m2scorer', 'errant'], help='target filepath')
    args = parser.parse_args()
    main(args)
