import getting_data
import os
import sys
import numpy as np
from process import s2search_score_ale
from process import s2search_score_pipelining
from process import s2search_score_shap
from process import s2search_score_anchor
import plotting_nb_gen
import ranker_helper
import shapley_value

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


class XaiProcess:

    def __init__(self, exp_name) -> None:
        self.work_dir = os.path.dirname(os.path.abspath(__file__))
        self.exp_dir_path = os.path.join(self.work_dir, 'pipelining', exp_name)

        scores_dir = os.path.join(self.exp_dir_path, 'scores')
        if not os.path.exists(scores_dir):
            os.mkdir(scores_dir)

        config = getting_data.read_conf(self.exp_dir_path)
        description, sample_configs, samples_from_other_exp_configs = config

        self.description = description
        self.sample_configs = sample_configs
        self.samples_from_other_exp_configs = samples_from_other_exp_configs

        self.exp_name = exp_name

    def get_sample_config_for_this_method(self, sample_name, method):
        ranker_helper.set_ranker_logger(self.exp_dir_path, method)
        if (type(self.sample_configs[sample_name]) == dict):
            # new config
            return self.sample_configs[sample_name][method]
        else:
            return self.sample_configs[sample_name]

    def get_data_info(self, sample_name):
        if self.samples_from_other_exp_configs != None and sample_name in self.samples_from_other_exp_configs.keys():
            other_exp_name, data_file_name = self.samples_from_other_exp_configs.get(
                sample_name)
            return {
                'sample_src_exp_name': other_exp_name,
                'current_sample_name': sample_name,
                'sample_src_name': data_file_name.replace('.data', '')
            }
        else:
            return {
                'sample_src_exp_name': self.exp_name,
                'current_sample_name': sample_name,
                'sample_src_name': sample_name
            }

    def ale(self, is_2w=False):
        for sample_name in self.sample_configs.keys():
            sample_config = self.get_sample_config_for_this_method(sample_name,
                                                                   'ale_2w' if is_2w else 'ale_1w')
            s2search_score_ale.get_ale_and_save_score(
                self.exp_dir_path, is_2w, sample_config, self.get_data_info(sample_name))
            plotting_nb_gen.gen_for_ale(
                self.exp_name, self.description, [sample_name])

    def masking(self):
        for sample_name in self.sample_configs.keys():
            sample_config = self.get_sample_config_for_this_method(
                sample_name, 'masking')
            for t in sample_config:
                t['masking_option_keys'] = getting_data.masking_option_keys.copy()
            s2search_score_pipelining.masking_score(os.path.join(
                self.work_dir, 'pipelining'), self.exp_name, self.get_data_info(sample_name), sample_config)

    def shapley_value(self):
        self.masking()
        for sample_name in self.sample_configs.keys():
            plotting_nb_gen.gen_for_shapley_value(
                self.exp_name, self.description, [sample_name])
            shapley_value.compute_shapley_value(self.exp_name, sample_name)

    def check_query(self):
        def _percentage(count, total):
            return f'{count}({round(count / total * 100, 2)}%)'
        for sample_name in self.sample_configs.keys():
            sample_config = self.get_sample_config_for_this_method(
                sample_name, 'check_query')
            paper_data = getting_data.load_sample(
                self.get_data_info(sample_name)['sample_src_exp_name'], self.get_data_info(sample_name)['sample_src_name'], not_df=True)
            paper_data_len = len(paper_data)

            for query in sample_config:
                ranker_helper.start_record_paper_count(
                    f'check query {self.exp_name} {sample_name}')
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
    distribution of {self.get_data_info(sample_name)['current_sample_name']} is: A: {_percentage(cluster_a_count, paper_data_len)}, \
    B: {_percentage(cluster_b_count, paper_data_len)}, C: {_percentage(cluster_c_count, paper_data_len)}")

                ranker_helper.end_record_paper_count(
                    f'check query {self.exp_name} {sample_name}')

    def samplining_shap(self, task_number):
        for sample_name in self.sample_configs.keys():
            sample_config = self.get_sample_config_for_this_method(
                sample_name, 'smpshap')
            if sample_config.get('query') != None and sample_config.get('task') != None:
                query = sample_config['query']
                task_config = sample_config['task'][task_number]

                s2search_score_shap.get_sampling_shap_shapley_value(
                    self.exp_dir_path, query, task_config, self.get_data_info(sample_name))

    def anchor(self, task_number):
        for sample_name in self.sample_configs.keys():
            sample_config = self.get_sample_config_for_this_method(
                sample_name, 'anchor')
            query = sample_config['query']
            explainer_configs = sample_config['explainer_configs']
            task_config = sample_config['task'][task_number]

            s2search_score_anchor.get_anchor_metrics(
                self.exp_dir_path, query, task_config, explainer_configs, self.get_data_info(sample_name))

    def remove_empty_log(self):
        log_dir = os.path.join(self.exp_dir_path, 'log')
        file_list = os.listdir(log_dir)
        for f in file_list:
            size = os.path.getsize(os.path.join(log_dir, f))
            if size == 0:
                os.remove(os.path.join(log_dir, f))


def get_smp_shap_data(exp_name, sample_name_list=None):
    work_dir = os.path.dirname(os.path.abspath(__file__))
    exp_dir_path = os.path.join(work_dir, 'pipelining', exp_name)
    config = getting_data.read_conf(exp_dir_path)
    description, sample_configs, samples_from_other_exp_configs = config

    shap_data = {}
    check_list = sample_configs.keys() if sample_name_list == None else sample_name_list
    for sample_name in check_list:
        sample_smshap_config = sample_configs[sample_name]['smpshap']
        if sample_smshap_config.get('task') != None:
            shap_sv = []
            shap_bv = []
            for task in sample_smshap_config.get('task'):
                rg = task['range']
                m_file = os.path.join(
                    exp_dir_path, 'scores', f"{sample_name}_shap_sampling_{rg[0]}_{rg[1]}.npz")
                if os.path.exists(m_file):
                    ld = np.load(m_file)

                    shap_values = ld['shap_values']
                    base_values = ld['base_values']

                    shap_sv.extend(shap_values)
                    shap_bv.extend(base_values)

            shap_data[sample_name] = dict(
                shap_sv=shap_sv,
                shap_bv=shap_bv,
            )
        else:
            src_exp_name, src_sample_name = sample_smshap_config = sample_configs[
                sample_name]['smpshap']['data_from']
            src_shap_data = get_smp_shap_data(src_exp_name, [src_sample_name])
            shap_data[sample_name] = src_shap_data[src_sample_name]

    return shap_data


if __name__ == '__main__':
    if len(sys.argv) > 1:
        exp_list = [md
                    for md in sys.argv[1:] if not md.startswith('--') and md not in [str(n) for n in range(20)]]

        task_number_from_arg = [md
                                for md in sys.argv[1:] if md in [str(n) for n in range(20)]]

        method = [md.replace('--', '')
                  for md in sys.argv if md.startswith('--')][0]

        for exp_name in exp_list:
            if method not in method_key:
                print(f'no such method: {method}')
            else:
                xps = XaiProcess(exp_name)
                if method == 'ale_1w':
                    xps.ale(is_2w=False)
                if method == 'ale_2w':
                    xps.ale(is_2w=True)
                if method == 'masking':
                    xps.masking()
                if method == 'sv':
                    xps.shapley_value()
                if method == 'smpshap':
                    task_number = int(task_number_from_arg[0])
                    xps.samplining_shap(task_number)
                if method == 'check_query':
                    xps.check_query()
                if method == 'anchor':
                    task_number = int(task_number_from_arg[0])
                    xps.anchor(task_number)
                xps.remove_empty_log()
