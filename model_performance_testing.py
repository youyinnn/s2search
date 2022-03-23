from dis import dis
from s2search_score_pipelining import get_scores
import json
import time
import sys

if __name__ == '__main__':
    multiple = 1
    if len(sys.argv) > 1:
        multiple = int(sys.argv[1])
    paper = []
    with open('./test.data') as f:
        for l in f.readlines():
            for i in range(multiple):
                paper.append(json.loads(l.strip()))

    # st = time.time()
    # scores_nw = get_scores('Machine Learning', paper, task_name="Noworker", force_global=True)
    # print(f"compute {len(paper)} paper scores within {round(time.time() - st, 6)} sec")

    print(len(paper))
    
    st = time.time()
    scores_w = get_scores('Machine Learning', paper, task_name="Worker", force_global=False)
    print(f"compute {len(paper)} paper scores within {round(time.time() - st, 6)} sec")
    
    # same = True
    # for i in range(len(scores_nw)):
    #     if scores_nw[i] != scores_w[i]:
    #         same = False

    # print(f'the result with or without worker is the same: {same}')