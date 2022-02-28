from s2search_score_pipelining import get_scores
import json
import time

if __name__ == '__main__':
    paper = []
    with open('./test.data') as f:
        for l in f.readlines():
            paper.append(json.loads(l.strip()))

    st = time.time()
    scores_nw = get_scores('Machine Learning', paper, use_pool=False, task_name="Noworker")
    print(f"compute {len(paper)} paper scores within {round(time.time() - st, 6)} sec")
    
    print()
    
    st = time.time()
    scores_w = get_scores('Machine Learning', paper, use_pool=True, task_name="Worker")
    print(f"compute {len(paper)} paper scores within {round(time.time() - st, 6)} sec")
    
    diff = False
    for i in range(len(scores_nw)):
        if scores_nw[i] != scores_w[i]:
            diff = True

    print(f'the result with or without worker is the same: {diff}')