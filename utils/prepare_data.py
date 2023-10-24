import argparse
import functools
import math
import json
import random

from functools import partial
from itertools import combinations
from multiprocessing import Pool

from edit import (
    read_m2,
    m2_to_edits,
    sort_edits,
    sort_edits_no_type,
    edits_to_text,
    filter_conflict,
    powerset,
)
from edit import _IGNORE_TYPE, _EDIT_START, _EDIT_END, _EDIT_TYPE, _EDIT_COR

from tqdm import tqdm
import errant
errant = errant.load('en')


def generate_label(datum, relabel_edit=True, merge_consecutive=True, with_gap=True):
    final_data = []
    result = {}
    source = datum['source']
    src_len = len(source.split())
    targets = datum.get('refs', None)

    if 'hyp' in datum:
        hyp = datum['hyp']
    elif 'edits' in datum:
        edits = datum['edits']
        hyp, _ = edits_to_text(source, edits)
    else:
        raise ValueError("each datum should contain either data/hyp key")
    hyp_len = len(hyp.strip().split())

    result['source'] = source.split()
    result['hyp'] = hyp.split()

    src_mask = [False] * src_len # 0 means ignore
    hyp_mask = [False] * hyp_len # 0 means ignore
    if with_gap:
        gap_mask = [False] * (hyp_len + 1)
    
    f_half = -1
    if relabel_edit:
        orig = errant.parse(source)
        hyp_parse = errant.parse(hyp)
        hyp_alignment = errant.align(orig, hyp_parse)
        hyp_token_edits = errant.merge(hyp_alignment, merging='all-split')

        for edit in hyp_token_edits:
            src_mask[edit.o_start:edit.o_end] = [True] * (edit.o_end - edit.o_start)
            hyp_mask[edit.c_start:edit.c_end] = [True] * (edit.c_end - edit.c_start)
            if with_gap and edit.c_end == edit.c_start:
                gap_mask[edit.c_start] = True
        
        assert len(src_mask) == src_len, "Length of mask ({}) does not match the words ({})" \
            .format(len(src_mask), src_len)
        assert len(hyp_mask) == hyp_len, "Length of mask ({}) does not match the words ({})" \
            .format(len(hyp_mask), hyp_len)
        result['masks']  = [src_mask, hyp_mask]

        if targets is not None:
            trg, _ = edits_to_text(source, targets)
            trg = errant.parse(trg)
            src_trg_alignment = errant.align(orig, trg)

            # create src_label
            src_trg_edits = errant.merge(src_trg_alignment, merging='all-split')
            
            src_label = [1] * src_len
            for edit in src_trg_edits:
                src_label[edit.o_start:edit.o_end] = [0] * (edit.o_end - edit.o_start)
            
            assert len(src_label) == src_len, "Length of label ({}) does not match the words ({})" \
                .format(len(src_label), src_len)              
            
            # create hyp_label
            hyp_trg_alignment = errant.align(hyp_parse, trg)
            hyp_trg_edits = errant.merge(hyp_trg_alignment, merging='all-split')
            hyp_label = [1] * hyp_len
            if with_gap:
                gap_label = [1] * (hyp_len + 1)
            for edit in hyp_trg_edits:
                if with_gap and edit.o_start == edit.o_end:
                    gap_label[edit.o_start] = 0
                hyp_label[edit.o_start:edit.o_end] = [0] * (edit.o_end - edit.o_start)

            assert len(hyp_label) == hyp_len, "Length of label ({}) does not match the words ({}) {}" \
                .format(len(hyp_label), hyp_len, 'label:\n{}\nhyp:\n{}'.format(hyp_label, hyp)) 

            # calculating M2 score
            if merge_consecutive:
                edits = errant.merge(hyp_alignment)
                edits = set([(e.o_start, e.o_end, e.c_str) for e in edits])
                targets = errant.merge(src_trg_alignment)
                targets = set([(t.o_start, t.o_end, t.c_str) for t in targets])

                tp = edits & targets
                fp = edits - tp
                fn = targets - edits
                
                precision = 1 if len(edits) == 0 else float(len(tp)) / len(edits)
                recall = 1 if len(targets) == 0 else float(len(tp)) / len(targets)
                f_half = 0 if precision + recall == 0 else \
                        (1 + 0.5 * 0.5) * precision * recall / (0.5 * 0.5 * precision + recall)
                if f_half == 0:
                    imprecision = 1 - precision
                    f_half = -0.1 * imprecision
            result['labels'] = [[round(f_half, 2)], src_label, hyp_label]
            if with_gap:
                result['gap'] = gap_label
                result['gap_mask'] = gap_mask
    else:
        raise NotImplementedError("There's a problem in getting the labels, " + \
            "so this is not implemented yet unless deemed necessary")
    return result


def hierarchical_generate_label(data, sort_labels=True, relabel_edit=True, merge_consecutive=True,
        with_gap=True, data_type='list'):
    label_fn = partial(generate_label, relabel_edit=relabel_edit,
                merge_consecutive=merge_consecutive, with_gap=with_gap)
    all_results = []
    for datum in tqdm(data):
        with Pool() as p:
            result = p.map(label_fn, datum)
            if sort_labels:
                result = sorted(result, reverse=True, key=lambda x:x['labels'][0][0])
            all_results.append(result)
    if data_type == 'json':
        formatted_result = []
        for batch in all_results:
            dict_result = {
                'source': [],
                'hyp': [],
                'masks': [],
                'gap': [],
                'gap_mask': [],
                'labels': []
            }
            for datum in batch:
                for k, v in datum.items():
                    if k not in dict_result:
                        print('[WARNING] creating new key', k)
                        dict_result[k] = []
                    dict_result[k].append(v)
            formatted_result.append(dict_result)
        all_results = formatted_result
    return all_results


def generate_unmodified_data(files, source, target, identity_percentage=0, bucket_size=-1,
        sample_hyp=False, keep_no_edit=True, keep_perfect=False, use_all=False):
    with open(source, encoding='utf-8') as f:
        sources = [l.strip() for l in f.readlines()]
    
    hyp_docs = []
    num_hyps = len(files)
    if bucket_size <= 0:
        bucket_size = num_hyps
    for hyp_path in files:
        with open(hyp_path, encoding='utf-8') as f:
            hyp_docs.append([l.strip() for l in f.readlines()])
    
    if target is not None:
        target_m2 = read_m2(target)
        trg_entities = []
        targets = []
        for m in target_m2:
            trg_ent = m2_to_edits(m)
            trg_edits = sort_edits(trg_ent['edits'])
            trg_entities.append(trg_edits)
            trg_sent, _ = edits_to_text(trg_ent['source'], trg_edits)
            targets.append(trg_sent)
    else:
        trg_entities = [None for _ in range(len(hyps_m2[0]))]
    
    corrected_data = [set([]) for _ in range(len(trg_entities))]
    for hyp_sents in hyp_docs:
        for s_idx, sent in enumerate(hyp_sents):
            sent = sent.strip()
            add_filter = corrected_data[s_idx].copy()
            if not keep_no_edit:
                add_filter.add(sources[s_idx])
            if not keep_perfect:
                add_filter.add(targets[s_idx])
            if sent.strip() not in add_filter:
                corrected_data[s_idx].add(sent)
    
    total_instance = sum([len(_set) for _set in corrected_data])
    all_identity_idx = [idx for idx, _set in enumerate(corrected_data) \
                        if len(_set) == 0]
    sample_size = total_instance * identity_percentage / (1 - identity_percentage)
    random.shuffle(all_identity_idx)
    identity_idx = all_identity_idx[:round(sample_size)]

    processed_data = []
    for s_idx, sent_set in enumerate(corrected_data):
        sent_set = list(sent_set)
        if len(sent_set) > bucket_size and sample_hyp:
            sent_set = random.sample(sent_set, bucket_size)
        else:
            while len(sent_set) > 0:
                instance_data = []
                for c_sent in sent_set[:bucket_size]:
                    instance_data.append({
                        'source': sources[s_idx],
                        'hyp': c_sent,
                        'refs': trg_entities[s_idx]
                    })
                if hierarchical:
                    processed_data.append(instance_data)
                else:
                    processed_data.extend(instance_data)
                sent_set = sent_set[bucket_size:]
    
    print('Mapping the data length...')
    assert hierarchical, "fill_in_grouping is only to be used for hierarchical processing"
    final_data = []
    len_map = {i: [] for i in range(1, bucket_size)}

    # mapping all instances' lengths
    instance_count = 0
    for s_idx, instance in enumerate(processed_data):
        ins_len = len(instance)
        if ins_len == bucket_size:
            final_data.append(instance)
        elif ins_len == 0:
            continue
        elif ins_len < bucket_size:
            len_map[ins_len].append(s_idx)
            instance_count += 1
        else:
            raise ValueError("There's a bug")
    print('Acquired {} full data'.format(len(final_data)))

    print('Grouping {} data...'.format(instance_count))
    print({k: len(v) for k, v in len_map.items()})
    used_identities = 0
    all_empty = False
    if use_all:
        min_bucket_hyp = 0
    else:
        min_bucket_hyp = (bucket_size // 2) - 1
    for cur_num_hyp in range(bucket_size - 1, 1, -1):
        while len(len_map[cur_num_hyp]) > 0 and not all_empty:
            instance_idx = len_map[cur_num_hyp].pop()
            instance = processed_data[instance_idx]
            
            check_empty = True
            member_len = len(instance)
            while(member_len < bucket_size) and not all_empty:
                filler_num = bucket_size - member_len
                found = False
                for i in range(filler_num):
                    if len(len_map[filler_num - i]) > 0 and not found:
                        add_idx = len_map[filler_num - i].pop()
                        instance.extend(processed_data[add_idx])
                        found = True
                    check_empty = check_empty and (len(len_map[filler_num - i]) == 0)
                if not found:
                    add_indices = all_identity_idx[:filler_num]
                    if len(add_indices) < filler_num:
                        print('[WARNING] Not enough filler, skipping.')
                        break
                    
                    all_identity_idx = all_identity_idx[filler_num:]
                    add_instances = [processed_data[s_i_i] for s_i_i in add_indices]
                    instance.extend(add_instances)
                    used_identities += len(add_indices)
                member_len = len(instance)
            
            all_empty = check_empty
            if member_len == bucket_size:
                final_data.append(instance)
            else:
                print('[WARNING] an instance with {} is skipped'.format(member_len))
    
    if used_identities < round(sample_size):
        print('[WARNING] only {} out of {} identity samples are used'\
                .format(used_identities, round(sample_size)))

    print('Generating labels...')
    result = hierarchical_generate_label(final_data)
    
    return result


def main(args):
    data = generate_unmodified_data(args.data, args.source_path, args.target_path,
                bucket_size=args.bucket_size, identity_percentage=args.identity_percentage,
                sample_hyp=args.sample_hyp, keep_no_edit=args.keep_no_edit, keep_perfect=args.keep_perfect,
                use_all=args.use_all)
    
    with open(args.output_path, 'w', encoding='utf-8') as out:
        for datum in data:
            out.write(json.dumps(datum))
            out.write('\n')


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', nargs='+', default=[])
    parser.add_argument('--mix_edits', default=False, action='store_true', help='mix the edits')
    parser.add_argument('--target_path', default=None, help='path to the target file during data generation')
    parser.add_argument('--source_path', default=None, help='path to the source file during data generation')
    parser.add_argument('--output_path', default='out.txt', help='path to the output file during testing')
    parser.add_argument('--identity_percentage', type=float, default=0.15, help="percentage of identity data included")
    parser.add_argument('--bucket_size', type=int, default=4, help="size of data group to be ranked during training")
    parser.add_argument('--merge_consecutive', default=True, action='store_true', help='merge consecutive edits')
    parser.add_argument('--teacher_force', default=False, action='store_true', help='include new instance with correct edit contexts')
    parser.add_argument('--include_source', default=False, action='store_true', help='include new instance without any edits')
    parser.add_argument('--fast', default=False, action='store_true', help='use fast generation for h_mix')
    parser.add_argument('--sample_hyp', default=False, action='store_true', help='Sample when number of hypotheses is bigger than bucket size')
    parser.add_argument('--keep_no_edit', default=False, action='store_true', help='Include hypotheses that is similar to the source sentence')
    parser.add_argument('--keep_perfect', default=False, action='store_true', help='Include hypotheses that is similar to the target sentence')
    parser.add_argument('--use_all', default=False, action='store_true', help='Use all hypotheses even when the variants are limited')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()
    main(args)