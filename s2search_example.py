import json
import urllib.request
from s2search.rank import S2Ranker

data_dir = './s2search_data'

papers_example = [
    {
        'title': 'Jumping NLP Curves: A Review of Natural Language Processing Research',
        'abstract': 'Natural language processing (NLP) is a theory-motivated range of computational techniques for the automatic analysis and representation of human language. NLP research has evolved from the era of punch cards and batch processing (in which the analysis of a sentence could take up to 7 minutes) to the era of Google and the likes of it (in which millions of webpages can be processed in less than a second). This review paper draws on recent developments in NLP research to look at the past, present, and future of NLP technology in a new light. Borrowing the paradigm of jumping curves from the field of business management and marketing prediction, this survey article reinterprets the evolution of NLP research as the intersection of three overlapping curves-namely Syntactics, Semantics, and Pragmatics Curveswhich will eventually lead NLP research to evolve into natural language understanding.',
        'venue': 'IEEE Computational intelligence ',
        'authors': ['E Cambria', 'B White'],
        'year': 2014,
        'n_citations': 900,
    }
]


def S2_Rank(related_keywords, paper_dict_list, file=data_dir):
    s2ranker = S2Ranker(file)
    score = s2ranker.score(related_keywords, paper_dict_list)
    return score


print(S2_Rank('NLP', papers_example, data_dir))
