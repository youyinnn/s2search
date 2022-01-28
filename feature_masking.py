import copy, json
import add_on

masking_options = {
    't': {
        'plot_legend': 'title',
        'should_mask': ['title'],
        'marker': '.',
        'color': 'r'
    }, 
    'abs': {
        'plot_legend': 'abstract',
        'should_mask':['abstract'],
        'marker': ',',
        'color': 'g'
    }, 
    'v': {
        'plot_legend': 'venue',
        'should_mask':['venue'],
        'marker': 'v',
        'color': 'y'
    }, 
    'au': {
        'plot_legend': 'authors',
        'should_mask':['authors'],
        'marker': 'o',
        'color': 'b'
    }, 
    'y': {
        'plot_legend': 'year',
        'should_mask':['year'],
        'marker': 'x',
        'color': 'm'
    }, 
    'c': {
        'plot_legend': 'n_citations',
        'should_mask':['n_citations'],
        'marker': '*',
        'color': 'c'
    },
    'rt1': {
        'plot_legend': 'replaceing title1',
        # 'should_mask': ['title'],
        'should_replace': ['title'],
        'marker': '<',
        'color': 'black',
        'replace_func': add_on.replace_title_1
    },
    'rt2': {
        'plot_legend': 'replaceing title2',
        # 'should_mask': ['title'],
        'should_replace': ['title'],
        'marker': '>',
        'color': 'gold',
        'replace_func': add_on.replace_title_2
    }
    # 'all': ['title', 'abstract', 'venue', 'authors', 'year','n_citations'],
}

def masking_with_option(original_paper_data, options):
    cp = copy.deepcopy(original_paper_data)
    for paper in cp:
        if options.get('should_mask') is not None:
            for masking_feature in options['should_mask']:        
                if masking_feature == 'authors':
                    paper['authors'] = []

                # same as del paper['n_citations']
                elif masking_feature == 'n_citations':
                    paper['n_citations'] = 0       

                # same as paper['year'] = ""
                # elif masking_feature == 'year':       
                #     del paper['year']

                else:
                    paper[masking_feature] = " "
        if options.get('should_replace') is not None:
            for replace_feature in options['should_replace']:
                paper[replace_feature] = options['replace_func'](paper[replace_feature])
    return cp


def masking(original_paper_data, masking_option_keys = ["t", "abs", "v", "au", "y", "c", 'rt1']):
    all_result = {}
    for key in masking_option_keys:
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
        print()
