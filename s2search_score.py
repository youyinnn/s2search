from s2search.rank import S2Ranker
import os
import json
import numpy as np
from pathlib import Path

# data_dir = './s2search_data'
s2_dir = './s2search_data'
root_dir = '/Users/yinnnyou/workspace/XAI_PROJECT/data_process/masking'
features = ['title', 'abstract', 'venue', 'authors', 'year', 'n_citations', 'full']
papers_example = [
    {
        'title': 'Jumping NLP Curves: A Review of Natural Language Processing Research',
        'abstract': 'Natural language processing (NLP) is a theory-motivated range of computational techniques for '
                    'the automatic analysis and representation of human language. NLP research has evolved from the '
                    'era of punch cards and batch processing (in which the analysis of a sentence could take up to 7 '
                    'minutes) to the era of Google and the likes of it (in which millions of webpages can be '
                    'processed in less than a second). This review paper draws on recent developments in NLP research '
                    'to look at the past, present, and future of NLP technology in a new light. Borrowing the '
                    'paradigm of jumping curves from the field of business management and marketing prediction, '
                    'this survey article reinterprets the evolution of NLP research as the intersection of three '
                    'overlapping curves-namely Syntactics, Semantics, and Pragmatics Curveswhich will eventually lead '
                    'NLP research to evolve into natural language understanding.',
        'venue': 'IEEE Computational intelligence ',
        'authors': ['E Cambria', 'B White'],
        'year': 2014,
        'n_citations': 900,
    }
]


def S2_Rank(related_keywords, paper_dict_list, file=s2_dir):
    s2ranker = S2Ranker(file)
    score = s2ranker.score(related_keywords, paper_dict_list)
    return score


def S2_open_json(path):
    data = []
    with open(path) as f:
        Lines = f.readlines()
        for line in Lines:
            line_strip = line.strip()
            jso = json.loads(line_strip, strict=False)
            data.append(jso)
    return S2_Rank('machine learning', data, s2_dir)


def S2_save_score_as_np(s2score, feature):
    base_dir = str(Path(__file__).resolve().parent)
    data_dir = os.path.join(base_dir)
    os.environ.setdefault("DATA_DIR", data_dir)
    output_data_file_name = os.path.join(os.environ.get("DATA_DIR"), "score" + feature)
    np.save(output_data_file_name, s2score)


def S2_get_score(root_dir):
    score = []
    for root, dirs, files in os.walk(root_dir):
        for name in files:
            if name.endswith((".json")):
                for feature in features:
                    if feature in name:
                        full_path = os.path.join(root, name)
                        print(full_path)
                        score = S2_open_json(full_path)
                        score = np.array(score)
                        print(score)
                        S2_save_score_as_np(score, feature)


S2_get_score(root_dir)
# print(S2_Rank('NLP', papers_example, s2_dir))
# score = np.load('/Users/ayuee/Documents/GitHub/XAI_PROJECT/data_process/masking/full_Score.npy')
# print(score, np.shape(score))
