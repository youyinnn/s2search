import os
import os.path as path
import yaml
import sys
import nbformat as nbf

data_dir = './pipelining'
user_repo = 'DingLi23'
branch = 'pipelining'
# user_repo = 'youyinnn'


def gen_for_normal(exp_name, description, sample_list):
    for sample_name in sample_list:
        p_nb_file = path.join(data_dir, exp_name,
                              f'{exp_name}_{sample_name}_plotting.ipynb')
        if (not path.exists(p_nb_file)):
            print(
                f'Generating plotting notebook for {exp_name} at {p_nb_file}')

            nb = nbf.v4.new_notebook()
            nb.metadata.kernelspec = {
                "display_name": "Python 3",
                "name": "python3"
            }
            nb.metadata.language_info = {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            }
            open_in_colab_href = f'<a href="https://colab.research.google.com/github/{user_repo}/s2search/blob/{branch}/pipelining/{exp_name}/{exp_name}_{sample_name}_plotting.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>'
            exp_des = f'### Experiment Description\n\n{description.strip()}\n\n> This notebook is for experiment \\<{exp_name}\\> and data sample \\<{sample_name}\\>.'

            init_md = '### Initialization'
            init_code = f'''%load_ext autoreload
%autoreload 2
import numpy as np, sys, os
in_colab = 'google.colab' in sys.modules
# fetching code and data(if you are using colab
if in_colab:
    !rm -rf s2search
    !git clone --branch pipelining https://github.com/youyinnn/s2search.git
    sys.path.insert(1, './s2search')
    %cd s2search/pipelining/{exp_name}/

pic_dir = os.path.join('.', 'plot')
if not os.path.exists(pic_dir):
    os.mkdir(pic_dir)
'''

            loading_data_md = '### Loading data'
            loading_data_code = f'''sys.path.insert(1, '../../')
import numpy as np, sys, os
from getting_data import get
from feature_masking import masking_options

sample_data_and_config_arr = get('{exp_name}', '{sample_name}')

for sample_data_and_config in sample_data_and_config_arr:
    y_values = []
    sample_origin_npy = sample_data_and_config['origin']
    for array in sample_data_and_config['feature_stack']:

        # define your y axis value here
        y_value = np.absolute((sample_origin_npy - array) / sample_origin_npy)
        # y_value = sample_origin_npy - array
        y_values.append(y_value)

    y_values = np.array(y_values)
    sample_data_and_config['y_values'] = y_values
'''

            plot_data_md = "### Plot the data"
            plot_data_code = '''import matplotlib.pyplot as plt
def plot_scores_d(sample_name, y_values, sample_origin_npy, query, sample_masking_option_keys, sample_task_number): 
  plt.figure(figsize=(20, 15), dpi=80)
  i = 0
  for key in sample_masking_option_keys:
    plt.scatter(
        sample_origin_npy,
        y_values[i],
        c=masking_options[key]['color'], 
        marker=masking_options[key]['marker'],
        label=masking_options[key]['plot_legend']
    )
    i += 1

  plt.xlabel('Orginal Ranking Score',fontsize=16)
  plt.ylabel('$y = \\\\frac{|Score_0 - Score\_feature_j|}{Score_0}$', fontsize=16)
  plt.title(f'Distrubution of Ranking Score and Masking Features Difference\\nfor \\'{sample_name}\\', with query \\'{query}\\'', fontsize=20, pad=20)
  x_max = 10
  x_min = 0
  x_pace = 1
  y_max = 10
  y_min = 0
  y_pace = 1
#   plt.xticks(np.arange(x_min, x_max, x_pace), size = 9) 
#   plt.yticks(np.arange(y_min, y_max, y_pace), size = 9)
#   plt.ylim(y_min, y_max)
#   plt.xlim(x_min, x_max)
  plt.legend(prop={'size': 16})
  plt.savefig(os.path.join('.', 'plot', f'{sample_name}-t{sample_task_number}.png'), facecolor='white', transparent=False)
  plt.show()

for sample_data_and_config in sample_data_and_config_arr:
  sample_and_task_name = sample_data_and_config['sample_and_task_name']
  sample_origin_npy = sample_data_and_config['origin']
  y_values = sample_data_and_config['y_values']

  valid_data_count = 0
  total_data_count = 0
  for d in sample_origin_npy:
    total_data_count += 1
    if d > -10:
      valid_data_count += 1
  print(f'({valid_data_count}/{total_data_count}) ({round((valid_data_count / total_data_count) * 100, 2)}%) valid data for {sample_and_task_name} (original score is greater than -10)')

  sample_query = sample_data_and_config['query']
  sample_masking_option_keys = sample_data_and_config['masking_option_keys']
  sample_task_number = sample_data_and_config['task_number']
  plot_scores_d(sample_and_task_name, y_values, sample_origin_npy, sample_query, sample_masking_option_keys, sample_task_number)

'''

            nb['cells'] = [
                nbf.v4.new_markdown_cell(open_in_colab_href),
                nbf.v4.new_markdown_cell(exp_des),

                nbf.v4.new_markdown_cell(init_md),
                nbf.v4.new_code_cell(init_code),

                nbf.v4.new_markdown_cell(loading_data_md),
                nbf.v4.new_code_cell(loading_data_code),

                nbf.v4.new_markdown_cell(plot_data_md),
                nbf.v4.new_code_cell(plot_data_code),
            ]
            nbf.write(nb, str(p_nb_file))


def gen_for_ale(exp_name, description, sample_list):
    for sample_name in sample_list:
        ale_1w_score_file_name = f'{sample_name}_1w_ale_title.npz'
        ale_1w_score_file = path.join(data_dir, exp_name, 'scores', ale_1w_score_file_name)
        if (path.exists(ale_1w_score_file)):
            ale_1w_nb_file_name = f'{exp_name}_{sample_name}_1w_ale_plotting.ipynb'
            ale_1w_nb_file = path.join(data_dir, exp_name, ale_1w_nb_file_name)
            if (not path.exists(ale_1w_nb_file)):
                print(
                    f'Generating plotting notebook for {exp_name} at {ale_1w_nb_file}')

                nb = nbf.v4.new_notebook()
                nb.metadata.kernelspec = {
                    "display_name": "Python 3",
                    "name": "python3"
                }
                nb.metadata.language_info = {
                    "codemirror_mode": {
                        "name": "ipython",
                        "version": 3
                    },
                    "file_extension": ".py",
                    "mimetype": "text/x-python",
                    "name": "python",
                    "nbconvert_exporter": "python",
                    "pygments_lexer": "ipython3",
                }
                open_in_colab_href = f'<a href="https://colab.research.google.com/github/{user_repo}/s2search/blob/{branch}/pipelining/{exp_name}/{ale_1w_nb_file_name}" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>'
                exp_des = f'### Experiment Description\n\n{description.strip()}\n\n> This notebook is for experiment \\<{exp_name}\\> and data sample \\<{sample_name}\\>.'

                init_md = '### Initialization'
                init_code = f'''%load_ext autoreload
%autoreload 2
import numpy as np, sys, os
in_colab = 'google.colab' in sys.modules
# fetching code and data(if you are using colab
if in_colab:
    !rm -rf s2search
    !git clone --branch pipelining https://github.com/youyinnn/s2search.git
    sys.path.insert(1, './s2search')
    %cd s2search/pipelining/{exp_name}/

pic_dir = os.path.join('.', 'plot')
if not os.path.exists(pic_dir):
    os.mkdir(pic_dir)
'''

                loading_data_md = '### Loading data'
                loading_data_code = f'''
sys.path.insert(1, '../../')
import numpy as np, sys, os, pandas as pd
from getting_data import read_conf
from s2search_score_pdp import pdp_based_importance

sample_name = '{sample_name}'

f_list = [
    'title', 'abstract', 'venue', 'authors', 
    'year', 
    'n_citations'
    ]
ale_xy = {{}}
ale_metric = pd.DataFrame(columns=['feature_name', 'ale_range', 'ale_importance'])

for f in f_list:
    file = os.path.join('.', 'scores', f'{{sample_name}}_1w_ale_{{f}}.npz')
    if os.path.exists(file):
        nparr = np.load(file)
        quantile = nparr['quantile']
        ale_result = nparr['ale_result']
        values_for_rug = nparr.get('values_for_rug')
        
        ale_xy[f] = {{
            'x': quantile,
            'y': ale_result,
            'rug': values_for_rug,
            'weird': ale_result[len(ale_result) - 1] > 20
        }}
        
        if f != 'year' and f != 'n_citations':
            ale_xy[f]['x'] = list(range(len(quantile)))
            ale_xy[f]['numerical'] = False
        else:
            ale_xy[f]['xticks'] = quantile
            ale_xy[f]['numerical'] = True
            
        ale_metric.loc[len(ale_metric.index)] = [f, np.max(ale_result) - np.min(ale_result), pdp_based_importance(ale_result, f)]
               
        # print(len(ale_result))
        
print(ale_metric.sort_values(by=['ale_importance'], ascending=False))
print()
        
des, sample_configs, sample_from_other_exp = read_conf('.')
if sample_configs['{sample_name}'][0].get('quantiles') != None:
    print(f'The following feature choose quantiles as ale bin size:')
    for k in sample_configs['{sample_name}'][0]['quantiles'].keys():
        print(f"\t{{k}} with {{sample_configs['{sample_name}'][0]['quantiles'][k]}}% quantile, {{len(ale_xy[k]['x'])}} bins are used")
if sample_configs['{sample_name}'][0].get('intervals') != None:
    print(f'The following feature choose fixed amount as ale bin size:')
    for k in sample_configs['{sample_name}'][0]['intervals'].keys():
        print(f"\t{{k}} with {{sample_configs['{sample_name}'][0]['intervals'][k]}} values, {{len(ale_xy[k]['x'])}} bins are used")
'''

                plot_data_md = "### ALE Plots"
                plot_data_code = '''import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

categorical_plot_conf = [
    {
        'xlabel': 'Title',
        'ylabel': 'ALE',
        'ale_xy': ale_xy['title']
    },
    {
        'xlabel': 'Abstract',
        'ale_xy': ale_xy['abstract']
    },    
    {
        'xlabel': 'Authors',
        'ale_xy': ale_xy['authors'],
        # 'zoom': {
        #     'inset_axes': [0.3, 0.3, 0.47, 0.47],
        #     'x_limit': [92, 95],
        #     'y_limit': [-0.005, 0.03],
        # }
    },    
    {
        'xlabel': 'Venue',
        'ale_xy': ale_xy['venue'],
        # 'zoom': {
        #     'inset_axes': [0.3, 0.3, 0.47, 0.47],
        #     'x_limit': [89, 94],
        #     'y_limit': [-1, 6],
        # }
    },
]

numerical_plot_conf = [
    {
        'xlabel': 'Year',
        'ylabel': 'ALE',
        'ale_xy': ale_xy['year'],
        'rug': True
    },
    {
        'xlabel': 'Citations',
        'ale_xy': ale_xy['n_citations'],
        # 'zoom': {
        #     'inset_axes': [0.4, 0.2, 0.47, 0.47],
        #     'x_limit': [-1000.0, 12000],
        #     'y_limit': [-0.1, 1.2],
        # },
    },
]

def pdp_plot(confs, title):
    fig, axes_list = plt.subplots(nrows=1, ncols=len(confs), figsize=(20, 5), dpi=100)
    subplot_idx = 0
    plt.suptitle(title, fontsize=20, fontweight='bold')
    # plt.autoscale(False)
    for conf in confs:
        axes = axes if len(confs) == 1 else axes_list[subplot_idx]
        
        sns.rugplot(conf['ale_xy']['rug'], ax=axes, height=0.02)

        axes.plot(conf['ale_xy']['x'], conf['ale_xy']['y'])
        axes.grid(alpha = 0.4)

        # axes.set_ylim([-2, 20])
        axes.xaxis.set_major_locator(MaxNLocator(integer=True))
        axes.yaxis.set_major_locator(MaxNLocator(integer=True))
        
        if ('ylabel' in conf):
            axes.set_ylabel(conf.get('ylabel'), fontsize=20, labelpad=10)
        
        # if ('xticks' not in conf['ale_xy'].keys()):
        #     xAxis.set_ticklabels([])

        axes.set_xlabel(conf['xlabel'], fontsize=16, labelpad=10)
        
        if not (conf['ale_xy']['weird']):
            if (conf['ale_xy']['numerical']):
                # axes.set_ylim([-1, 3])
                pass
            else:
                axes.set_ylim([-2, 20])
                
        if 'zoom' in conf:
            axins = axes.inset_axes(conf['zoom']['inset_axes']) 
            axins.xaxis.set_major_locator(MaxNLocator(integer=True))
            axins.yaxis.set_major_locator(MaxNLocator(integer=True))
            axins.plot(conf['ale_xy']['x'], conf['ale_xy']['y'])
            axins.set_xlim(conf['zoom']['x_limit'])
            axins.set_ylim(conf['zoom']['y_limit'])
            axins.grid(alpha=0.3)
            rectpatch, connects = axes.indicate_inset_zoom(axins)
            connects[0].set_visible(False)
            connects[1].set_visible(False)
            connects[2].set_visible(True)
            connects[3].set_visible(True)
            
        subplot_idx += 1

pdp_plot(categorical_plot_conf, f"ALE for {len(categorical_plot_conf)} categorical features")
plt.savefig(os.path.join('.', 'plot', f'{sample_name}-1wale-categorical.png'), facecolor='white', transparent=False, bbox_inches='tight')

pdp_plot(numerical_plot_conf, f"ALE for {len(numerical_plot_conf)} numerical features")
plt.savefig(os.path.join('.', 'plot', f'{sample_name}-1wale-numerical.png'), facecolor='white', transparent=False, bbox_inches='tight')

plt.show()
'''

                nb['cells'] = [
                    nbf.v4.new_markdown_cell(open_in_colab_href),
                    nbf.v4.new_markdown_cell(exp_des),

                    nbf.v4.new_markdown_cell(init_md),
                    nbf.v4.new_code_cell(init_code),

                    nbf.v4.new_markdown_cell(loading_data_md),
                    nbf.v4.new_code_cell(loading_data_code),

                    nbf.v4.new_markdown_cell(plot_data_md),
                    nbf.v4.new_code_cell(plot_data_code),
                ]
                nbf.write(nb, str(ale_1w_nb_file))
        else:
            print(f'no 1-way score file for {sample_name}')
            
        ale_2w_score_file_name = f'{sample_name}_2w_ale_title_abstract.npz'
        ale_2w_score_file = path.join(data_dir, exp_name, 'scores', ale_2w_score_file_name)
        if (path.exists(ale_2w_score_file)):
            ale_2w_nb_file_name = f'{exp_name}_{sample_name}_2w_ale_plotting.ipynb'
            ale_2w_nb_file = path.join(data_dir, exp_name, ale_2w_nb_file_name)
            if (not path.exists(ale_2w_nb_file)):
                print(
                    f'Generating plotting notebook for {exp_name} at {ale_2w_nb_file}')

                nb = nbf.v4.new_notebook()
                nb.metadata.kernelspec = {
                    "display_name": "Python 3",
                    "name": "python3"
                }
                nb.metadata.language_info = {
                    "codemirror_mode": {
                        "name": "ipython",
                        "version": 3
                    },
                    "file_extension": ".py",
                    "mimetype": "text/x-python",
                    "name": "python",
                    "nbconvert_exporter": "python",
                    "pygments_lexer": "ipython3",
                }
                open_in_colab_href = f'<a href="https://colab.research.google.com/github/{user_repo}/s2search/blob/{branch}/pipelining/{exp_name}/{ale_2w_nb_file_name}" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>'
                exp_des = f'### Experiment Description\n\n{description.strip()}\n\n> This notebook is for experiment \\<{exp_name}\\> and data sample \\<{sample_name}\\>.'

                init_md = '### Initialization'
                init_code = f'''%load_ext autoreload
%autoreload 2
import numpy as np, sys, os
in_colab = 'google.colab' in sys.modules
# fetching code and data(if you are using colab
if in_colab:
    !rm -rf s2search
    !git clone --branch pipelining https://github.com/youyinnn/s2search.git
    sys.path.insert(1, './s2search')
    %cd s2search/pipelining/{exp_name}/

pic_dir = os.path.join('.', 'plot')
if not os.path.exists(pic_dir):
    os.mkdir(pic_dir)
'''

                loading_data_md = '### Loading data'
                loading_data_code = f'''

sys.path.insert(1, '../../')
import numpy as np, sys, os, pandas as pd
from getting_data import read_conf
pd.options.display.float_format = '{{:,.8f}}'.format

sample_name = '{sample_name}'

f_list = [
    'title', 
    'abstract', 
    'venue', 
    # 'authors', 
    'year', 
    'n_citations'
    ]
ale_rs = []
ale_metric = pd.DataFrame(columns=['f1_name', 'f2_name', '2w_ale_range', 'mean', 'var', 'std'])

def replace_quantile(feature_name, quantile):
    if feature_name == 'year' or feature_name == 'n_citations':
        return quantile
    else:
        return list(range(len(quantile)))

for i in range(len(f_list)):
    f1_name = f_list[i]
    for j in range(i + 1, len(f_list)):
        f2_name = f_list[j]
        file = os.path.join('.', 'scores', f'{{sample_name}}_2w_ale_{{f1_name}}_{{f2_name}}.npz')
        if os.path.exists(file):
            nparr = np.load(file)
            quantile_1 = nparr['quantile_1']
            quantile_2 = nparr['quantile_2']
            ale_result = nparr['ale_result']
            
            if np.mean(ale_result) != 0:
                norm = np.linalg.norm(ale_result)
                ale_result = ale_result / norm
            
            t = f'The mean of the 2-way ale - ({{f1_name}} * {{f2_name}}): {{np.mean(ale_result)}}'
            
            ale_metric.loc[len(ale_metric.index)] = \
                [f1_name, f2_name, np.max(ale_result) - np.min(ale_result), np.mean(ale_result),\
                    np.var(ale_result, ddof=1),np.std(ale_result, ddof=1)]
            
            ale = {{
                'ale': ale_result,
                'f1_quantile': replace_quantile(f1_name, quantile_1),
                'f2_quantile': replace_quantile(f2_name, quantile_2),
                'f1_name': f1_name,
                'f2_name': f2_name,
                'title': t
            }}
            
            ale_rs.append(ale)

print(ale_metric)
'''

                plot_data_md = "### ALE Plots"
                plot_data_code = '''import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

for ale in ale_rs:
    fig, ax = plt.subplots(figsize=(14, 10), constrained_layout=True)
    cmap = plt.colormaps['BuGn']
    
    im = ax.pcolormesh(ale['f1_quantile'], ale['f2_quantile'], ale['ale'],  cmap=None, edgecolors='k', linewidths=0)

    # cf = ax.contourf(ale['f1_quantile'], ale['f2_quantile'], ale['ale'], levels=50, alpha=0.7)
    
    ax.set_xlabel(ale['f1_name'], fontsize=16, labelpad=10)
    ax.set_ylabel(ale['f2_name'], fontsize=16, labelpad=10)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title(ale['title'], fontsize=20, pad=10)
    fig.colorbar(im, ax=ax, shrink=0.6, pad=0.03)
    plt.savefig(os.path.join('.', 'plot', f"{sample_name}-2wale-{ale['f1_name']}-{ale['f2_name']}.png"), facecolor='white', transparent=False, bbox_inches='tight')
'''

                nb['cells'] = [
                    nbf.v4.new_markdown_cell(open_in_colab_href),
                    nbf.v4.new_markdown_cell(exp_des),

                    nbf.v4.new_markdown_cell(init_md),
                    nbf.v4.new_code_cell(init_code),

                    nbf.v4.new_markdown_cell(loading_data_md),
                    nbf.v4.new_code_cell(loading_data_code),

                    nbf.v4.new_markdown_cell(plot_data_md),
                    nbf.v4.new_code_cell(plot_data_code),
                ]
                nbf.write(nb, str(ale_2w_nb_file))
        else:
            print(f'no 2-way score file for {sample_name}')


def gen_for_pdp(exp_name, description, sample_list):
    for sample_name in sample_list:
        p_nb_file = path.join(data_dir, exp_name,
                              f'{exp_name}_{sample_name}_plotting.ipynb')
        if (not path.exists(p_nb_file)):
            print(
                f'Generating plotting notebook for {exp_name} at {p_nb_file}')

            nb = nbf.v4.new_notebook()
            nb.metadata.kernelspec = {
                "display_name": "Python 3",
                "name": "python3"
            }
            nb.metadata.language_info = {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.9.7"
            }
            open_in_colab_href = f'<a href="https://colab.research.google.com/github/{user_repo}/s2search/blob/{branch}/pipelining/{exp_name}/{exp_name}_{sample_name}_plotting.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>'
            exp_des = f'### Experiment Description\n\n{description.strip()}\n\n> This notebook is for experiment \\<{exp_name}\\> and data sample \\<{sample_name}\\>.'

            init_md = '### Initialization'
            init_code = f'''%load_ext autoreload
%autoreload 2
import numpy as np, sys, os
in_colab = 'google.colab' in sys.modules
# fetching code and data(if you are using colab
if in_colab:
    !rm -rf s2search
    !git clone --branch pipelining https://github.com/youyinnn/s2search.git
    sys.path.insert(1, './s2search')
    %cd s2search/pipelining/{exp_name}/

pic_dir = os.path.join('.', 'plot')
if not os.path.exists(pic_dir):
    os.mkdir(pic_dir)
'''

            loading_data_md = '### Loading data'
            loading_data_code = f'''sys.path.insert(1, '../../')
import numpy as np, sys, os, pandas as pd
from s2search_score_pdp import pdp_based_importance, apply_order

sample_name = '{sample_name}'

f_list = ['title', 'abstract', 'venue', 'authors', 'year', 'n_citations']

pdp_xy = {{}}
pdp_metric = pd.DataFrame(columns=['feature_name', 'pdp_range', 'pdp_importance'])

for f in f_list:
    file = os.path.join('.', 'scores', f'{{sample_name}}_pdp_{{f}}.npz')
    if os.path.exists(file):
        data = np.load(file)
        sorted_pdp_data = apply_order(data)
        feature_pdp_data = [np.mean(pdps) for pdps in sorted_pdp_data]
        
        pdp_xy[f] = {{
            'y': feature_pdp_data,
            'numerical': True
        }}
        if f == 'year' or f == 'n_citations':
            pdp_xy[f]['x'] = np.sort(data['arr_1'])
        else:
            pdp_xy[f]['y'] = feature_pdp_data
            pdp_xy[f]['x'] = list(range(len(feature_pdp_data)))
            pdp_xy[f]['numerical'] = False
            
        pdp_metric.loc[len(pdp_metric.index)] = [f, np.max(feature_pdp_data) - np.min(feature_pdp_data), pdp_based_importance(feature_pdp_data, f)]
            
        pdp_xy[f]['weird'] = feature_pdp_data[len(feature_pdp_data) - 1] > 30
        

print(pdp_metric.sort_values(by=['pdp_importance'], ascending=False))
'''

            plot_data_md = "### PDP"
            plot_data_code = '''import matplotlib.pyplot as plt

categorical_plot_conf = [
    {
        'xlabel': 'Title',
        'ylabel': 'Scores',
        'pdp_xy': pdp_xy['title']
    },
    {
        'xlabel': 'Abstract',
        'pdp_xy': pdp_xy['abstract']
    },    
    {
        'xlabel': 'Authors',
        'pdp_xy': pdp_xy['authors']
    },
    {
        'xlabel': 'Venue',
        'pdp_xy': pdp_xy['venue'],
        # 'zoom': {
        #     'inset_axes': [0.15, 0.45, 0.47, 0.47],
        #     'x_limit': [950, 1010],
        #     'y_limit': [-9, 7],
        #     'connects': [True, True, False, False]
        # }
    },
]

numerical_plot_conf = [
    {
        'xlabel': 'Year',
        'ylabel': 'Scores',
        'pdp_xy': pdp_xy['year']
    },
    {
        'xlabel': 'Citation Count',
        'pdp_xy': pdp_xy['n_citations'],
        # 'zoom': {
        #     'inset_axes': [0.5, 0.2, 0.47, 0.47],
        #     'x_limit': [-100, 1000],
        #     'y_limit': [-7.3, -6.2],
        #     'connects': [False, False, True, True]
        # }
    }
]

def pdp_plot(confs, title):
    fig, axes = plt.subplots(nrows=1, ncols=len(confs), figsize=(20, 5), dpi=100)
    subplot_idx = 0
    plt.suptitle(title, fontsize=20, fontweight='bold')
    # plt.autoscale(False)
    for conf in confs:
        axess = axes if len(confs) == 1 else axes[subplot_idx]

        axess.plot(conf['pdp_xy']['x'], conf['pdp_xy']['y'])
        axess.grid(alpha = 0.4)

        if ('ylabel' in conf):
            axess.set_ylabel(conf.get('ylabel'), fontsize=20, labelpad=10)
        
        axess.set_xlabel(conf['xlabel'], fontsize=16, labelpad=10)
        
        if not (conf['pdp_xy']['weird']):
            if (conf['pdp_xy']['numerical']):
                # axess.set_ylim([-1, 3])
                pass
            else:
                axess.set_ylim([-15, 10])
                pass
                
        if 'zoom' in conf:
            axins = axess.inset_axes(conf['zoom']['inset_axes']) 
            axins.plot(conf['pdp_xy']['x'], conf['pdp_xy']['y'])
            axins.set_xlim(conf['zoom']['x_limit'])
            axins.set_ylim(conf['zoom']['y_limit'])
            axins.grid(alpha=0.3)
            rectpatch, connects = axess.indicate_inset_zoom(axins)
            connects[0].set_visible(conf['zoom']['connects'][0])
            connects[1].set_visible(conf['zoom']['connects'][1])
            connects[2].set_visible(conf['zoom']['connects'][2])
            connects[3].set_visible(conf['zoom']['connects'][3])
            
        subplot_idx += 1

pdp_plot(categorical_plot_conf, "PDPs for four categorical features")
plt.savefig(os.path.join('.', 'plot', f'{sample_name}-categorical.png'), facecolor='white', transparent=False, bbox_inches='tight')

# second fig
pdp_plot(numerical_plot_conf, "PDPs for two numerical features")
plt.savefig(os.path.join('.', 'plot', f'{sample_name}-numerical.png'), facecolor='white', transparent=False, bbox_inches='tight')
'''

            nb['cells'] = [
                nbf.v4.new_markdown_cell(open_in_colab_href),
                nbf.v4.new_markdown_cell(exp_des),

                nbf.v4.new_markdown_cell(init_md),
                nbf.v4.new_code_cell(init_code),

                nbf.v4.new_markdown_cell(loading_data_md),
                nbf.v4.new_code_cell(loading_data_code),

                nbf.v4.new_markdown_cell(plot_data_md),
                nbf.v4.new_code_cell(plot_data_code),
            ]
            nbf.write(nb, str(p_nb_file))

        ice_nb_file = path.join(data_dir, exp_name,
                              f'{exp_name}_ice_{sample_name}_plotting.ipynb')
        if (not path.exists(ice_nb_file)):
            print(f'Generating plotting notebook for {exp_name} at {ice_nb_file}')
        
            nb = nbf.v4.new_notebook()
            nb.metadata.kernelspec = {
                "display_name": "Python 3",
                "name": "python3"
            }
            nb.metadata.language_info = {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.9.7"
            }
            open_in_colab_href = f'<a href="https://colab.research.google.com/github/{user_repo}/s2search/blob/{branch}/pipelining/{exp_name}/{exp_name}_ice_{sample_name}_plotting.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>'
            exp_des = f'### Experiment Description\n\n{description.strip()}\n\n> This notebook is for experiment \\<{exp_name}\\> and data sample \\<{sample_name}\\>.'

            init_md = '### Initialization'
            init_code = f'''%load_ext autoreload
%autoreload 2
import numpy as np, sys, os
in_colab = 'google.colab' in sys.modules
# fetching code and data(if you are using colab
if in_colab:
    !rm -rf s2search
    !git clone --branch pipelining https://github.com/youyinnn/s2search.git
    sys.path.insert(1, './s2search')
    %cd s2search/pipelining/{exp_name}/

pic_dir = os.path.join('.', 'plot')
if not os.path.exists(pic_dir):
    os.mkdir(pic_dir)
'''

            loading_data_md = '### Loading data'
            loading_data_code = f'''sys.path.insert(1, '../../')
import numpy as np, sys, os, pandas as pd
from s2search_score_pdp import pdp_based_importance, apply_order

sample_name = '{sample_name}'

f_list = ['title', 
          'abstract', 'venue', 'authors', 'year', 'n_citations']

pdp_xy = {{}}

for f in f_list:
    file = os.path.join('.', 'scores', f'{{sample_name}}_pdp_{{f}}.npz')
    if os.path.exists(file):
        data = np.load(file)
        sorted_pdp_data = apply_order(data)
        average_pdp = [np.mean(pdps) for pdps in sorted_pdp_data]
        
        # tran = sorted_pdp_data
        tran = np.flipud(np.rot90(sorted_pdp_data))
        
        average_pdp -= average_pdp[0]
        
        for instance in tran:
            anchor = instance[0]
            instance -= anchor
        
        pdp_xy[f] = {{
            'y': tran,
            'numerical': True,
            'average': average_pdp
        }}
        if f == 'year' or f == 'n_citations':
            pdp_xy[f]['x'] = np.sort(data['arr_1'])
        else:
            pdp_xy[f]['x'] = list(range(len(sorted_pdp_data)))
            pdp_xy[f]['numerical'] = False
'''

            plot_data_md = "### PDP"
            plot_data_code = '''import matplotlib.pyplot as plt

categorical_plot_conf = [
    {
        'xlabel': 'Title',
        'ylabel': 'Scores',
        'pdp_xy': pdp_xy['title']
    },
    {
        'xlabel': 'Abstract',
        'pdp_xy': pdp_xy['abstract']
    },    
    {
        'xlabel': 'Authors',
        'pdp_xy': pdp_xy['authors']
    },
    {
        'xlabel': 'Venue',
        'pdp_xy': pdp_xy['venue'],
        # 'zoom': {
        #     'inset_axes': [0.15, 0.45, 0.47, 0.47],
        #     'x_limit': [950, 1010],
        #     'y_limit': [-9, 7],
        #     'connects': [True, True, False, False]
        # }
    },
]

numerical_plot_conf = [
    {
        'xlabel': 'Year',
        'ylabel': 'Scores',
        'pdp_xy': pdp_xy['year']
    },
    {
        'xlabel': 'Citation Count',
        'pdp_xy': pdp_xy['n_citations'],
        # 'zoom': {
        #     'inset_axes': [0.5, 0.2, 0.47, 0.47],
        #     'x_limit': [-100, 1000],
        #     'y_limit': [-7.3, -6.2],
        #     'connects': [False, False, True, True]
        # }
    }
]

def pdp_plot(confs, title):
    fig, axes = plt.subplots(nrows=1, ncols=len(confs), figsize=(20, 5), dpi=100)
    subplot_idx = 0
    plt.suptitle(title, fontsize=20, fontweight='bold')
    # plt.autoscale(False)
    c = '#AAAAAA'
    for conf in confs:
        axess = axes if len(confs) == 1 else axes[subplot_idx]

        axess.plot(conf['pdp_xy']['x'], conf['pdp_xy']['average'], c='#FFF301', lw=2, zorder=2)

        for y in conf['pdp_xy']['y'][:]:
            axess.plot(conf['pdp_xy']['x'], y, 
                       c=c,zorder=1
            )
        axess.grid(alpha = 0.4)

        if ('ylabel' in conf):
            axess.set_ylabel(conf.get('ylabel'), fontsize=20, labelpad=10)
        
        axess.set_xlabel(conf['xlabel'], fontsize=16, labelpad=10)
        
        if (conf['pdp_xy']['numerical']):
            axess.set_ylim([-0.1, 2.5])
            pass
        else:
            axess.set_ylim([-1, 27])
            pass
                
        if 'zoom' in conf:
            axins = axess.inset_axes(conf['zoom']['inset_axes']) 
            axins.plot(conf['pdp_xy']['x'], conf['pdp_xy']['average'], c='#FFF301', lw=2, zorder=2)
            for y in conf['pdp_xy']['y'][:]:
                axins.plot(conf['pdp_xy']['x'], y, 
                        c=c,zorder=1
                )
            axins.set_xlim(conf['zoom']['x_limit'])
            axins.set_ylim(conf['zoom']['y_limit'])
            axins.grid(alpha=0.3)
            rectpatch, connects = axess.indicate_inset_zoom(axins)
            connects[0].set_visible(conf['zoom']['connects'][0])
            connects[1].set_visible(conf['zoom']['connects'][1])
            connects[2].set_visible(conf['zoom']['connects'][2])
            connects[3].set_visible(conf['zoom']['connects'][3])
            
        subplot_idx += 1

pdp_plot(categorical_plot_conf, "ICE Plots for four categorical features")

# second fig
pdp_plot(numerical_plot_conf, "ICE Plots for two numerical features")
'''

            nb['cells'] = [
                nbf.v4.new_markdown_cell(open_in_colab_href),
                nbf.v4.new_markdown_cell(exp_des),

                nbf.v4.new_markdown_cell(init_md),
                nbf.v4.new_code_cell(init_code),

                nbf.v4.new_markdown_cell(loading_data_md),
                nbf.v4.new_code_cell(loading_data_code),

                nbf.v4.new_markdown_cell(plot_data_md),
                nbf.v4.new_code_cell(plot_data_code),
            ]
            nbf.write(nb, str(ice_nb_file))

if __name__ == '__main__':
    if len(sys.argv) >= 2:
        exp_name = sys.argv[1]
        exp_dir = path.join(data_dir, exp_name)
        if not os.path.exists(exp_dir):
            print(f'No {exp_name} experiment')
            exit()

        conf_path = path.join(exp_dir, 'conf.yml')
        with open(conf_path, 'r') as f:
            conf = yaml.safe_load(f)
            description = conf.get('description')
            sample_configs = conf.get('samples')
            sample_list = sample_configs.keys()

        # generate the plotting.ipynb if not exist
        if exp_name.startswith('exp'):
            gen_for_normal(exp_name, description, sample_list)

        if exp_name.startswith('pdp'):
            gen_for_pdp(exp_name, description, sample_list)
            
        if exp_name.startswith('ale'):
            gen_for_ale(exp_name, description, sample_list)
