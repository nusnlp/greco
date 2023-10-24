import argparse
import os
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

    all_data = []
    for s_id, src in enumerate(sources):
        s_data = {
            'src': src.strip(),
            'hyp': {},
        }
        for system_name in systems.keys():
            s_data['hyp'][system_name] = systems[system_name][s_id].strip()
        all_data.append(s_data)

    
    count_annots = 0
    count_sents = 0
    for data in all_data:
        if 'ranks' in data:
            count_annots += len(data['ranks'])
            count_sents += 1
        else:
            hyp_set = {}
            for system_name, hyp in data['hyp'].items():
                if hyp not in hyp_set:
                    hyp_set[hyp] = [system_name]
                else:
                    hyp_set[hyp].append(system_name)
            data['hyp_order'] = list(hyp_set.values())

        data['scores'] = [-2 for _ in data['hyp_order']]
    
    # create queue to get the LM score
    queue = {
        'src': [],
        'hyp': [],
        's_ids': [], # sentence id
        'h_ids': [], # hyp id within annotation
    }
    for s_id, data in enumerate(all_data):
        num_hyp = len(data['hyp_order'])
        for h_id, system_names in enumerate(data['hyp_order']):
            queue['s_ids'].append(s_id)
            queue['h_ids'].append(h_id)
            queue['src'].append(data['src'])
            queue['hyp'].append(data['hyp'][system_names[0]])
   
    data_len = len(queue['hyp'])
    print('Total hypotheses: ', data_len)
    model = get_model(args)
    model.eval()

    num_iter = (data_len + args.batch_size - 1) // args.batch_size
    for i in tqdm(range(num_iter)):
        # get the QE score
        b_start = i * args.batch_size
        b_end = min(b_start + args.batch_size, data_len)
        src = queue['src'][b_start:b_end]
        hyp = queue['hyp'][b_start:b_end]
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
            if isinstance(sc, list) and len(sc) == 1:
                sc = sc[0]
            assert math.isnan(sc) or sc >= 0, "all QE score should be >= 0"
            if math.isnan(sc):
                scores[sc_id] = -1

        # record the predicted rank
        s_ids = queue['s_ids'][b_start:b_end]
        h_ids = queue['h_ids'][b_start:b_end]
        assert len(scores) == len(s_ids) == len(h_ids)
        for s_id, h_id, score in zip(s_ids, h_ids, scores):
            all_data[s_id]['scores'][h_id] = score

    outputs = [s.strip() for s in sources]
    
    for s_id, sen_dict in enumerate(all_data):
        sys_names = sen_dict['hyp_order']
        scores = sen_dict['scores']
        best = sorted(zip(scores, sys_names), reverse=True, key=lambda x: x[0])
        best_sys = best[0][1][0] # first list item, second tuple item, first hyp name
        outputs[s_id] = sen_dict['hyp'][best_sys]

    with open(args.output_path, 'w', encoding='utf-8') as out:
        out.write('\n'.join(outputs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='XML file')
    parser.add_argument('--source_file', type=str, default=None, help='Optional source file')
    parser.add_argument('--output_path', type=str, help='Output path')
    parser.add_argument('--batch_size', type=int, default=240, help="batch size")
    parser.add_argument('--src_name', default='INPUT', type=str, help='source name')
    parser.add_argument('--model', default='gpt-2', help='scorer name')
    parser.add_argument('--lm_model', default=None, help='LM model name')
    parser.add_argument('--checkpoint', default=None, help="path to the model's checkpoint")
    parser.add_argument('--auto', default=False, action='store_true', help='read all files in the directory')
    parser.add_argument('--verbose', default=False, action='store_true', help='verbose')
    args = parser.parse_args()
    main(args)
