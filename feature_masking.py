import copy, json
import add_on

t = 'title'
abs = 'abstract'
v = 'venue'
y = 'year'
au = 'authors'
c = 'n_citations'

mp = {
    't': t,
    'abs': abs,
    'v': v,
    'y': y,
    'au': au,
    'c': c,
}

masking_options = {
    't': {
        'plot_legend': t,
        'should_mask': [t],
        'marker': '.',
        'color': 'r'
    }, 
    'abs': {
        'plot_legend': abs,
        'should_mask':[abs],
        'marker': ',',
        'color': 'g'
    }, 
    'v': {
        'plot_legend': v,
        'should_mask':[v],
        'marker': 'v',
        'color': 'y'
    }, 
    'au': {
        'plot_legend': au,
        'should_mask':[au],
        'marker': 'o',
        'color': 'b'
    }, 
    'y': {
        'plot_legend': y,
        'should_mask':[y],
        'marker': 'x',
        'color': 'm'
    }, 
    'c': {
        'plot_legend': c,
        'should_mask':[c],
        'marker': '*',
        'color': 'c'
    },
    'rt1': {
        'plot_legend': 'replaceing title1',
        # 'should_mask': [t],
        'should_replace': [t],
        'marker': '<',
        'color': 'black',
        'replace_func': add_on.replace_title_1
    },
    'rt2': {
        'plot_legend': 'replaceing title2',
        # 'should_mask': [t],
        'should_replace': [t],
        'marker': '>',
        'color': 'gold',
        'replace_func': add_on.replace_title_2
    },
}

def combination():
    def backtracing(_set, tmp, comb, start):
        if len(tmp) > 0:
            comb.append(copy.deepcopy(tmp))
        i = start
        while i < len(_set):
            tmp.append(_set[i])
            backtracing(_set, tmp, comb, i + 1)
            tmp.pop(len(tmp) - 1)
            i += 1

    _set = ['t', 'abs', 'v', 'au', 'y', 'c']
    comb = []
    backtracing(_set, [], comb, 0)
    return comb

def get_comb_masking_options():
    comb = combination()
    comb_map = {}
    color_range = [
        'dimgray', 
        'saddlebrown', 
        'deeppink', 

        'red', 
        'gold',
        'olivedrab', 

        'darkgreen', 
        'darkorange', 
        'cyan', 

        'navy',
        'purple', 
        'teal', 
    ]
    marker_range = [
        '1', '2', '3', '4',
        'x', '*', '+'
    ]
    ci = 0
    mi = 0
    for cb in comb:
        key = ''.join(cb)
        plot_legend = ''
        should_mask = []
        for k in cb:
            plot_legend += k + ' & '
            should_mask.append(mp[k])

        if ci == len(color_range):
            ci = 0

        color = color_range[ci]
        ci += 1

        if mi == len(marker_range):
            mi = 0

        marker = marker_range[mi]
        mi += 1

        comb_map[key] = {
            'plot_legend': plot_legend[:len(plot_legend) - 3],
            'should_mask': should_mask,
            'marker': marker,
            'color': color
        }
    
    return comb_map

comb_map = get_comb_masking_options()
for key in comb_map:
    if masking_options.get(key) == None:
        masking_options[key] = comb_map[key]

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
    # ars = masking(papers)
    # for key in ars.keys():
    #     print(f'masking {masking_options[key]}')
    #     print(json.dumps(ars[key]))
    #     print()

    for k in masking_options:
        print(k, masking_options[k].get('should_mask'), masking_options[k]['color'], masking_options[k]['plot_legend'] )