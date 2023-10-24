from itertools import chain, combinations
import random
from os.path import basename, isdir, isfile, join, splitext

_IGNORE_TYPE = {"noop", "UNK", "Um"}
_EDIT_START = 0
_EDIT_END = 1
_EDIT_TYPE = 2
_EDIT_COR = 3


def multiple_insertion(x, y):
    return x[0] == x[1] == y[0] == y[1]

def intersecting_range(x, y):
    return (x[0] <= y[0] < x[1] and not x[0] == y[1]) or \
            (y[0] <= x[0] < y[1] and not y[0] == x[1])

def no_conflict(edit, selected_edits):
    for selected_edit in selected_edits:
        if multiple_insertion(edit, selected_edit) \
            or intersecting_range(edit, selected_edit):
            return False
    
    return True

def filter_conflict(edits):
    filtered_edits = []
    for edit in edits:
        if no_conflict(edit, filtered_edits):
            filtered_edits.append(edit)
    return sorted(filtered_edits)

def powerset(iterable, shuffle=False):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    edit_nums = list(range(len(s)+1))
    if shuffle:
        random.shuffle(edit_nums)
    return chain.from_iterable(combinations(s, r) for r in edit_nums)

def m2_to_edits(m2_entity):
    m2_lines = m2_entity.split('\n')
    source = m2_lines[0][2:]
    edits = []
    for m2_line in m2_lines[1:]:
        if not m2_line.startswith("A"):
            raise ValueError("{} is not an m2 edit".format(m2_line))
        m2_line = m2_line[2:]
        features = m2_line.split("|||")
        span = features[0].split()
        start, end = int(span[0]), int(span[1])
        error_type = features[1]
        if error_type.strip() in _IGNORE_TYPE:
            continue
        replace_token = features[2]
        edits.append((start, end, error_type, replace_token))
    return {'source': source, 'edits': edits}

def read_m2(filepath):
    with open(filepath, encoding='utf-8') as f:
        m2_entries = f.read().strip().split('\n\n')
    
    return m2_entries

def read_data(src_path, file_path, m2_dir, target_m2=None, filter_idx=None):
    m2_path = join(m2_dir, splitext(basename(file_path))[0] + '.m2')

    if not isfile(m2_path):
        parse_m2(src_path, file_path, m2_path)
    
    hyp_m2 = read_m2(m2_path)
    if filter_idx is not None:
        hyp_m2 = [hyp_m2[i] for i in filter_idx]

    hyp_m2 = [m2_to_edits(m) for m in hyp_m2]

    if target_m2 is not None:
        assert len(target_m2) == len(hyp_m2), \
            "The m2 lengths of target ({}) and hypothesis ({}) are different!"\
                .format(len(target_m2), len(hyp_m2))
        for hyp_entry, trg_entry in zip(hyp_m2, target_m2):
            assert hyp_entry['source'] == trg_entry['source']
            hyp_edits = hyp_entry['edits']
            trg_edits = set([(t[_EDIT_START], t[_EDIT_END], t[_EDIT_COR]) for t in trg_entry['edits']])
            labels = []
            for edit in hyp_edits:
                e_start, e_end, e_type, e_cor = edit
                label = 1 if (e_start, e_end, e_cor) in trg_edits else 0
                labels.append(label)
            hyp_entry['labels'] = labels
    
    return hyp_m2

def sort_edits_no_type(edits, filter=True): # edits: [(start, end, error_type, replace_token)]
    if filter:
        edits = [e for e in edits if e[_EDIT_START] >= 0]
    edits = list(set(edits)) # remove duplicates
    return sorted(edits)

def sort_edits(edits, filter=True): # edits: [(start, end, error_type, replace_token)]
    if filter:
        edits = [e for e in edits if e[_EDIT_TYPE] not in _IGNORE_TYPE]
    edits = list(set(edits)) # remove duplicates
    return sorted(edits)

def edits_to_text(ori, edits, offset=0):
    _edits = sort_edits(edits)
    cor_sent = ori.split()

    offset = offset
    for edit in _edits:
        if edit[_EDIT_TYPE] in _IGNORE_TYPE: continue  # Ignore certain edits
        start = edit[_EDIT_START]
        end = edit[_EDIT_END]
        cor = edit[_EDIT_COR].split()
        len_cor = 0 if len(edit[_EDIT_COR]) == 0 else len(cor)
        cor_sent[start + offset:end + offset] = cor
        offset = offset - (end - start) + len_cor
    result = " ".join(cor_sent)

    return result, offset
