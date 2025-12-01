from s2search.rank import S2Ranker

# point to the artifacts downloaded from s3
data_dir = 's2search_data'

# the data is a list of dictionaries
papers = [
    {
        'title': 'Neural Networks are Great',
        'abstract': 'Neural networks are known to be really great models. You should use them.',
        'venue': 'Deep Learning Notions',
        'authors': ['Sergey Feldman', 'Gottfried W. Leibniz'],
        'year': 2019,
        'n_citations': 100,
        'n_key_citations': 10
    },
    {
        'title': 'Neural Networks are Terrible',
        'abstract': 'Neural networks have only barely worked and we should stop working on them.',
        'venue': 'JMLR',
        'authors': ['Isaac Newton', 'Sergey Feldman'],
        'year': 2009,
        'n_citations': 5000  # we don't have n_key_citations here and that's OK
    }
]

# only do this once because we have to load the giant language models into memory
s2ranker = S2Ranker(data_dir)

# higher scores are better
print(s2ranker.score('neural networks', papers))
print(s2ranker.score('feldman newton', papers))
print(s2ranker.score('jmlr', papers))
print(s2ranker.score('missing', papers))
