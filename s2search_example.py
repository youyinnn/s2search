import json
import urllib.request
from s2search.rank import S2Ranker

data_dir = './s2search_data'

papers_example = [
    # lower t
    {
        'title': 'A Panel Quantile Approach to Attrition Bias in Big Data: Evidence from a Randomized Experiment',
    },
    # upper t
    {
        'title': 'A Machine Learning guided Rewriting Approach for ASP Logic Programs',
    },    
    # lower abs
    {
        'abstract': 'This paper introduces a quantile regression estimator for panel data models with individual heterogeneity and attrition. The method is motivated by the fact that attrition bias is often encountered in Big Data applications. For example, many users sign-up for the latest program but few remain active users several months later, making the evaluation of such interventions inherently very challenging. Building on earlier work by Hausman and Wise (1979), we provide a simple identification strategy that leads to a two-step estimation procedure. In the first step, the coefficients of interest in the selection equation are consistently estimated using parametric or nonparametric methods. In the second step, standard panel quantile methods are employed on a subset of weighted observations. The estimator is computationally easy to implement in Big Data applications with a large number of subjects. We investigate the conditions under which the parameter estimator is asymptotically Gaussian and we carry out a series of Monte Carlo simulations to investigate the finite sample properties of the estimator. Lastly, using a simulation exercise, we apply the method to the evaluation of a recent Time-of-Day electricity pricing experiment inspired by the work of Aigner and Hausman (1980).',
    },    
    # upper abs
    {
        'abstract': 'Recently enacted legislation grants individuals certain rights to decide in what fashion their personal data may be used, and in particular a "right to be forgotten". This poses a challenge to machine learning: how to proceed when an individual retracts permission to use data which has been part of the training process of a model? From this question emerges the field of machine unlearning, which could be broadly described as the investigation of how to "delete training data from models". Our work complements this direction of research for the specific setting of class-wide deletion requests for classification models (e.g. deep neural networks). As a first step, we propose linear filtration as a intuitive, computationally efficient sanitization method. Our experiments demonstrate benefits in an adversarial setting over naive deletion schemes.',
    },
    
    # a
    {
        'title': 'A Panel Quantile Approach to Attrition Bias in Big Data: Evidence from a Randomized Experiment',
        'abstract': 'This paper introduces a quantile regression estimator for panel data models with individual heterogeneity and attrition. The method is motivated by the fact that attrition bias is often encountered in Big Data applications. For example, many users sign-up for the latest program but few remain active users several months later, making the evaluation of such interventions inherently very challenging. Building on earlier work by Hausman and Wise (1979), we provide a simple identification strategy that leads to a two-step estimation procedure. In the first step, the coefficients of interest in the selection equation are consistently estimated using parametric or nonparametric methods. In the second step, standard panel quantile methods are employed on a subset of weighted observations. The estimator is computationally easy to implement in Big Data applications with a large number of subjects. We investigate the conditions under which the parameter estimator is asymptotically Gaussian and we carry out a series of Monte Carlo simulations to investigate the finite sample properties of the estimator. Lastly, using a simulation exercise, we apply the method to the evaluation of a recent Time-of-Day electricity pricing experiment inspired by the work of Aigner and Hausman (1980).',
    },
    # b
    {
        'title': 'A Machine Learning guided Rewriting Approach for ASP Logic Programs',
        'abstract': 'This paper introduces a quantile regression estimator for panel data models with individual heterogeneity and attrition. The method is motivated by the fact that attrition bias is often encountered in Big Data applications. For example, many users sign-up for the latest program but few remain active users several months later, making the evaluation of such interventions inherently very challenging. Building on earlier work by Hausman and Wise (1979), we provide a simple identification strategy that leads to a two-step estimation procedure. In the first step, the coefficients of interest in the selection equation are consistently estimated using parametric or nonparametric methods. In the second step, standard panel quantile methods are employed on a subset of weighted observations. The estimator is computationally easy to implement in Big Data applications with a large number of subjects. We investigate the conditions under which the parameter estimator is asymptotically Gaussian and we carry out a series of Monte Carlo simulations to investigate the finite sample properties of the estimator. Lastly, using a simulation exercise, we apply the method to the evaluation of a recent Time-of-Day electricity pricing experiment inspired by the work of Aigner and Hausman (1980).',
    },
    # c
    {
        'title': 'A Panel Quantile Approach to Attrition Bias in Big Data: Evidence from a Randomized Experiment',
        'abstract': 'Recently enacted legislation grants individuals certain rights to decide in what fashion their personal data may be used, and in particular a "right to be forgotten". This poses a challenge to machine learning: how to proceed when an individual retracts permission to use data which has been part of the training process of a model? From this question emerges the field of machine unlearning, which could be broadly described as the investigation of how to "delete training data from models". Our work complements this direction of research for the specific setting of class-wide deletion requests for classification models (e.g. deep neural networks). As a first step, we propose linear filtration as a intuitive, computationally efficient sanitization method. Our experiments demonstrate benefits in an adversarial setting over naive deletion schemes.',
    },
    # d
    {
        'title': 'A Machine Learning guided Rewriting Approach for ASP Logic Programs',
        'abstract': 'Recently enacted legislation grants individuals certain rights to decide in what fashion their personal data may be used, and in particular a "right to be forgotten". This poses a challenge to machine learning: how to proceed when an individual retracts permission to use data which has been part of the training process of a model? From this question emerges the field of machine unlearning, which could be broadly described as the investigation of how to "delete training data from models". Our work complements this direction of research for the specific setting of class-wide deletion requests for classification models (e.g. deep neural networks). As a first step, we propose linear filtration as a intuitive, computationally efficient sanitization method. Our experiments demonstrate benefits in an adversarial setting over naive deletion schemes.',
    },
]

papers_example_year = [
]

for i in range(1900, 2031):
    papers_example_year.append({
        'year': i
    })
    
papers_example_ci = [
]

for i in range(0, 15001, 100):
    papers_example_ci.append({
        'n_citations': i
    })

import time
from s2search_score_pipelining import get_scores

st = time.time()

scores = get_scores('Machine Learning', papers_example)

et = round(time.time() - st, 6)

a,b,c,d = scores[len(scores) - 4: len(scores)]

print(scores)
print(a, b, c, d, (d - c) - (b - a))

print(f'{et} {len(papers_example)}')

scores = get_scores('Machine Learning', papers_example_year)

# print(scores)

scores = get_scores('Machine Learning', papers_example_ci)

print(scores)