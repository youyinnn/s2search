# Run Flask Demo

# python server_search.py

Make sure query_from_API(debug=True), if you don't want call s2search module which requires a larger complex env.
And you can Annotated from s2search.rank import S2Ranker in query_module.py.

# s2search

<details>
    <summary>Click to show</summary>

The Semantic Scholar Search Reranker

The code in this repo is for when you have a plain-text query and some academic documents,
and your goal is to search within the documents and obtain a score for how
good of a match each document is for the query. The standard pipeline involves a first-stage ranker (like ElasticSearch) and a reranker.
The model included with this repository is for the reranking stage only, but you may have few-enough documents
that a first-stage ranker is not necessary. The model and featurization are both fast.

## Installation

To install this package, run the following:

```bash
git clone https://github.com/allenai/s2search.git
cd s2search
conda create -y --name s2search python==3.7
conda activate s2search
python setup.py develop
pip install https://github.com/kpu/kenlm/archive/master.zip
```

To obtain the necessary data, run this command after the package is installed:

`aws s3 cp --no-sign-request s3://ai2-s2-research-public/s2search_data.zip .`

Then unzip the file. Iniside the zip is folder named `s2search/` that will contain all of the artifacts you'll need to get predictions.

Warning: this zip file is 10G compressed and 17G uncompressed.

## Example

Warning: you will need more than 17G of ram because of the large `kenlm` models that need to be loaded into memory.

An example of how to use this repo:

```python
from s2search.rank import S2Ranker

# point to the artifacts downloaded from s3
data_dir = 's2search/'

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
```

Note that `n_key_citations` is a Semantic Scholar feature. If you don't have it, just leave that key out of the data dictionary. The other paper fields are required.

</details>

# Score computing and plotting work flow

1. Create exp folder;
2. Put data into exp folder;
3. Create a config file for this experiment;
4. Run a command to use s2search to get the npy data;
5. Run a command to generate the jupyter notebook for data plotting of this experiment;

## Step 1-3. Setup experiment

1. Create a folder under `pipelining`, folder name would be the experiment name. Say `pipelining/exp4`.
2. Put all of your paper data under the experiment folder. Say `pipelining/exp4/cslg.data` and so on.
3. Create an experiment config file `conf.yml` under the experiment folder. Say `pipelining/exp4/conf.yml`.

   ```yaml
   description: "
   \nExperiment No.4, Plot all data"
   samples:
       cslg:
         - query: "Machine Learning"
           masking_option_keys: ["t", "abs", "v", "au", "y", "c"]
         - query: "convolutional neural network"
           masking_option_keys: ["t", "abs", "v", "au", "y", "c"]
       cscv:
         - query: "Computer Vision and Pattern Recognition"
           masking_option_keys: ["t", "abs", "v", "au", "y", "c"]
       csai:
         - query: "Artificial Intelligence"
           masking_option_keys: ["t", "abs", "v", "au", "y", "c"]
       csms:
         - query: "Mathematical Software"
           masking_option_keys: ["t", "abs", "v", "au", "y", "c"]
   ```

The configuration should contains those key-values.

- **description:** describe what this experiment is up to;
- **samples:** a **key-value map** of sample data: the key is the sample data file name, the value should contains **a list of task**, every task should have the following configs:
  - **query:** the query which input to the s2search;
  - **masking_option_keys:** a list of keys for masking option, this can be refer to [here](https://github.com/youyinnn/s2search/blob/85b3ac3e854b8903f92134d32515ae8313e3725e/feature_masking.py#L4);

## Step 4. Run s2search and get the npy files

```bash
python s2search_score_pipelining.py [experiment1_name] [experiment2_name] ...
```

E.g

```bash
python s2search_score_pipelining.py exp4
```

Then all npy data files will be created at `pipelining/exp1/scores` representing all the `masking_option_keys` that you config.

For instance, as for exp4 there should be four `.data` files under this experiment folder named as `cslg.data`, `cscv.daat` and so on. For `cslg.data`, the program will perform two tasks with different input queries then outputs the score files under `exp4/scores`.

You can also run multiple experiments:

```bash
python s2search_score_pipelining.py exp1 exp2 exp3
```

## Step 5. Generate plot notebook

Install nbformat frist:

```bash
pip install nbformat
```

You can:

```bash
python plotting_nb_gen.py [experiment_name(optional)] [sample_name(optional)]
```

For instance, when you do not have the `pipelining/exp4/exp4_csai_plotting.ipynb` for sample `csai` of `exp4`, run the command:

```bash
python plotting_nb_gen.py
```

This command will generate notebook files for those have not been created.

And the notebook will be generated will default setting.

Then you can check your plot with the notebook.

If you specify the exp name like:

```bash
python plotting_nb_gen.py exp1
```

It will force to generate all notebooks for all of the data samples of `exp1`.

If you specify the exp name like:

```bash
python plotting_nb_gen.py exp4 csai
```

It will force to generate the notebook for the data sample `csai` of `exp4`.
