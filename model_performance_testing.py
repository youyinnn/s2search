from dis import dis
from s2search_score_pipelining import get_scores, init_ranker
import json
import time

if __name__ == '__main__':
    paper = []
    with open('./test.data') as f:
        for l in f.readlines():
            paper.append(json.loads(l.strip()))

    st = time.time()
    scores_nw = get_scores('Machine Learning', paper, task_name="Noworker", use_pool=False)
    print(f"compute {len(paper)} paper scores within {round(time.time() - st, 6)} sec")

    print()
    
    st = time.time()
    scores_w = get_scores('Machine Learning', paper, task_name="Worker", use_pool=True)
    print(f"compute {len(paper)} paper scores within {round(time.time() - st, 6)} sec")
    
    same = True
    for i in range(len(scores_nw)):
        if scores_nw[i] != scores_w[i]:
            same = False

    print(f'the result with or without worker is the same: {same}')