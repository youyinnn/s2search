import copy, json

masking_options = {
    't': ['title'], 
    'abs': ['abstract'], 
    'v': ['venue'], 
    'au': ['authors'], 
    'y': ['year'], 
    'c': ['n_citations'],
    # 'all': ['title', 'abstract', 'venue', 'authors', 'year','n_citations'],
}

def masking_with_option(original_paper_data, options):
    cp = copy.deepcopy(original_paper_data)
    for paper in cp:
        for masking_feature in options:        
            if masking_feature == 'authors':
                paper['authors'] = []

            # same as del paper['n_citations']
            elif masking_feature == 'n_citations':
                paper['n_citations'] = 0       

            # same as paper['year'] = ""
            # elif masking_feature == 'year':       
            #     del paper['year']

            else:
                paper[masking_feature] = ""
    return cp


def masking(original_paper_data):
    all_result = {}
    for key in masking_options.keys():
        result = masking_with_option(original_paper_data, masking_options[key])
        all_result[key] = result
    return all_result

if __name__ == '__main__':
    # testing
    papers = [
        {
            'title': 'Neural Networks are Great',
            'abstract': 'Neural networks are known to be really great models. You should use them.',
            'venue': 'Deep Learning Notions',
            'authors': ['Sergey Feldman', 'Gottfried W. Leibniz'],
            'year': 2019,
            'n_citations': 100,
        },
        {
            'title': 'Neural Networks are Terrible',
            'abstract': 'Neural networks have only barely worked and we should stop working on them.',
            'venue': 'JMLR',
            'authors': ['Isaac Newton', 'Sergey Feldman'],
            'year': 2009,
            'n_citations': 5000
        }
    ]
    ars = masking(papers)
    for key in ars.keys():
        print(f'masking {masking_options[key]}')
        print(json.dumps(ars[key]))
