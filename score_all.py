import argparse
import os
import json
import math
import random

from tqdm import tqdm
from scipy.stats import rankdata

from models import get_model


_SYSTEMS = ['AMU', 'CAMB', 'CUUI', 'IITB', 'INPUT', 'IPN', 'NTHU', \
            'PKU', 'POST', 'RAC', 'SJTU', 'UFC', 'UMC']
# _SRC_NAME = 'INPUT'


def main(args):
    systems = {}
    if args.auto:
        for filepath in os.listdir(args.data_dir):
            system_name, _ = os.path.splitext(
                                os.path.basename(filepath))
            with open(os.path.join(args.data_dir, filepath)) as f:
                systems[system_name] = f.readlines()
    else:
        for system_name in _SYSTEMS:
            filepath = os.path.join(args.data_dir, system_name)
            with open(filepath) as f:
                systems[system_name] = f.readlines()

    if args.source_file is not None:
        with open(args.source_file) as f:
            systems[args.src_name] = f.readlines()
    sources = systems[args.src_name]
    
    model = get_model(args)
    model.eval()
    all_data = [{} for _ in sources]
    for system_name, hyps in systems.items():
        data_len = len(hyps)
        num_iter = (data_len + args.batch_size - 1) // args.batch_size
        for i in tqdm(range(num_iter)):
            # get the QE score
            b_start = i * args.batch_size
            b_end = min(b_start + args.batch_size, data_len)
            src = sources[b_start:b_end]
            hyp = hyps[b_start:b_end]
            scores = model.score(src, hyp)
            if not isinstance(scores, list):
                scores = scores.cpu().tolist()
            if not isinstance(scores, list):
                scores = [scores]
            assert len(scores) == len(hyp)
            if args.verbose:
                r_idx = random.randrange(b_end - b_start)
                print('===\nsrc: {}\nhyp: {}\nscore: {}\n'.format(
                    src[r_idx], hyp[r_idx], scores[r_idx]
                ))

            for sc_id, sc in enumerate(scores):
                if isinstance(sc, list):
                    sc = sc[0]
                assert math.isnan(sc) or sc >= 0, "all QE score should be >= 0"
                if math.isnan(sc):
                    scores[sc_id] = -1
                s_id = b_start + sc_id
                all_data[s_id][system_name] = {
                    'score': sc
                }
        
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
    parser.add_argument('--source_file', type=str, help='Path to source texts')
    parser.add_argument('--batch_size', type=int, default=16, help="batch size")
    parser.add_argument('--src_name', default='INPUT', type=str, help='source name')
    parser.add_argument('--model', default='greco', help='scorer name')
    parser.add_argument('--lm_model', default=None, help='LM model name')
    parser.add_argument('--checkpoint', default=None, help="path to the model's checkpoint")
    parser.add_argument('--auto', default=False, action='store_true', help='read all files in the directory')
    parser.add_argument('--verbose', default=False, action='store_true', help='verbose')
    args = parser.parse_args()
    main(args)
