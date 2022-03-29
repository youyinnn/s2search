import getting_data
import os
import sys
from process import s2search_score_ale
from process import s2search_score_pipelining
from process import s2search_score_shap
from process import s2search_score_anchor
import plotting_nb_gen
import ranker_helper

method_key = [
    # 'pdp_1w',
    # 'pdp_2w',
    'ale_1w',
    # 'ale_2w',
    # 'hs',
    'masking',
    'sv',
    'smpshap',
    'anchor',
    'check_query',
]

masking_option_keys = [
    "t",
    "abs",
    "v",
    "au",
    "y",
    "c",
    "tabs",
    "tabsv",
    "tabsvau",
    "tabsvauy",
    "tabsvauyc",
    "tabsvauc",
    "tabsvy",
    "tabsvyc",
    "tabsvc",
    "tabsau",
    "tabsauy",
    "tabsauyc",
    "tabsauc",
    "tabsy",
    "tabsyc",
    "tabsc",
    "tv",
    "tvau",
    "tvauy",
    "tvauyc",
    "tvauc",
    "tvy",
    "tvyc",
    "tvc",
    "tau",
    "tauy",
    "tauyc",
    "tauc",
    "ty",
    "tyc",
    "tc",
    "absv",
    "absvau",
    "absvauy",
    "absvauyc",
    "absvauc",
    "absvy",
    "absvyc",
    "absvc",
    "absau",
    "absauy",
    "absauyc",
    "absauc",
    "absy",
    "absyc",
    "absc",
    "vau",
    "vauy",
    "vauyc",
    "vauc",
    "vy",
    "vyc",
    "vc",
    "auy",
    "auyc",
    "auc",
    "yc",
]


class XaiProcess:

    def __init__(self, exp_name, sample_name) -> None:
        self.work_dir = os.path.dirname(os.path.abspath(__file__))
        self.exp_dir_path = os.path.join(self.work_dir, 'pipelining', exp_name)

        scores_dir = os.path.join(self.exp_dir_path, 'scores')
        if not os.path.exists(scores_dir):
            os.mkdir(scores_dir)

        config = getting_data.read_conf(self.exp_dir_path)
        description, sample_configs, samples_from_other_exp_configs = config

        self.description = description
        self.sample_configs = sample_configs

        if samples_from_other_exp_configs != None and sample_name in samples_from_other_exp_configs.keys():
            other_exp_name, data_file_name = samples_from_other_exp_configs.get(
                sample_name)
            self.data_info = {
                'sample_src_exp_name': other_exp_name,
                'current_sample_name': sample_name,
                'sample_src_name': data_file_name.replace('.data', '')
            }
        else:
            self.data_info = {
                'sample_src_exp_name': exp_name,
                'current_sample_name': sample_name,
                'sample_src_name': sample_name
            }

        self.exp_name = exp_name
        self.sample_name = sample_name

    def get_sample_config_for_this_method(self, method):
        ranker_helper.set_ranker_logger(self.exp_dir_path, method)
        if (type(self.sample_configs[self.sample_name]) == dict):
            # new config
            return self.sample_configs[self.sample_name][method]
        else:
            return self.sample_configs[self.sample_name]

    def ale(self, is_2w=False):
        sample_config = self.get_sample_config_for_this_method(
            'ale_2w' if is_2w else 'ale_1w')
        s2search_score_ale.get_ale_and_save_score(
            self.exp_dir_path, is_2w, sample_config, self.data_info)
        plotting_nb_gen.gen_for_ale(
            self.exp_name, self.description, [self.sample_name])

    def masking(self):
        sample_config = self.get_sample_config_for_this_method('masking')
        for t in sample_config:
            t['masking_option_keys'] = masking_option_keys
        s2search_score_pipelining.masking_score(os.path.join(
            self.work_dir, 'pipelining'), self.exp_name, self.data_info, sample_config)

    def shapley_value(self):
        self.masking()
        plotting_nb_gen.gen_for_shapley_value(
            self.exp_name, self.description, [self.sample_name])

    def check_query(self):
        def _percentage(count, total):
            return f'{count}({round(count / total * 100, 2)}%)'
        sample_config = self.get_sample_config_for_this_method('check_query')
        paper_data = getting_data.load_sample(
            self.data_info['sample_src_exp_name'], self.data_info['sample_src_name'], not_df=True)
        paper_data_len = len(paper_data)

        for query in sample_config:
            ranker_helper.start_record_paper_count(
                f'check query {self.exp_name} {self.sample_name}')
            scores = ranker_helper.get_scores(query, paper_data)
            cluster_a_count = 0
            cluster_b_count = 0
            cluster_c_count = 0
            for score in scores:
                if score < -10:
                    cluster_c_count += 1
                elif score < 0:
                    cluster_b_count += 1
                else:
                    cluster_a_count += 1

            print(f"for query \'{query}\', \
distribution of {self.data_info['current_sample_name']} is: A: {_percentage(cluster_a_count, paper_data_len)}, \
B: {_percentage(cluster_b_count, paper_data_len)}, C: {_percentage(cluster_c_count, paper_data_len)}")

            ranker_helper.end_record_paper_count(
                f'check query {self.exp_name} {self.sample_name}')

    def samplining_shap(self, task_number):
        sample_config = self.get_sample_config_for_this_method('smpshap')
        query = sample_config['query']
        task_config = sample_config['task'][task_number]

        s2search_score_shap.get_sampling_shap_shapley_value(
            self.exp_dir_path, query, task_config, self.data_info)

    def anchor(self, task_number):
        sample_config = self.get_sample_config_for_this_method('anchor')
        query = sample_config['query']
        explainer_configs = sample_config['explainer_configs']
        task_config = sample_config['task'][task_number]

        s2search_score_anchor.get_anchor_metrics(
            self.exp_dir_path, query, task_config, explainer_configs, self.data_info)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        exp_name = sys.argv[1]
        sample_name = sys.argv[2]
        method = sys.argv[3].replace('--', '')

        if method not in method_key:
            print(f'no such method: {method}')
        else:
            xps = XaiProcess(exp_name, sample_name)
            if method == 'ale_1w':
                xps.ale(is_2w=False)
            if method == 'ale_2w':
                xps.ale(is_2w=True)
            if method == 'masking':
                xps.masking()
            if method == 'sv':
                xps.shapley_value()
            if method == 'smpshap':
                task_number = int(sys.argv[4])
                xps.samplining_shap(task_number)
            if method == 'check_query':
                xps.check_query()
            if method == 'anchor':
                task_number = int(sys.argv[4])
                xps.anchor(task_number)
