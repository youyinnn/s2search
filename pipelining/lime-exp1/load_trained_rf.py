import numpy as np, sys, os, pandas as pd,json, os
sys.path.insert(1, '../../')
from getting_data import load_sample
from s2search_score_pipelining import init_ranker
import sklearn.ensemble
from sklearn.compose import ColumnTransformer
from joblib import dump, load

f_list = np.array([
    'title', 'abstract', 'venue', 'authors', 
    'year', 
    'n_citations'
])

exp_name = 'exp5'
sample_name = 'cslg'
query = 'Machine Learning'

if __name__ == '__main__':
    df = load_sample(exp_name, sample_name)
    paper_data = json.loads(df.to_json(orient='records'))

    data_in_arr = []

    for p in paper_data:
        p['authors'] = str(p['authors'])
        data_in_arr.append([p[feature_name] for feature_name in f_list])
        
    data_in_arr = np.array(data_in_arr, dtype='object')

    target_value_npz_file = os.path.join('.', 'scores', f'{sample_name}_target_value.npz')

    if os.path.exists(target_value_npz_file):
        target_value = np.load(target_value_npz_file)['arr_0']
    else:
        ranker = init_ranker()
        target_value = np.array(ranker.score(query, paper_data))
        
        scores_dir = os.path.join('.', 'scores')
        if not os.path.exists(str(scores_dir)):
            os.mkdir(str(scores_dir))
        print(f'\tsave PDP data to {target_value_npz_file}')
        np.savez_compressed(target_value_npz_file, target_value)


    categorical_features = np.array([0,1,2,3])

    le= sklearn.preprocessing.LabelEncoder()
    le.fit(target_value)
    labels = le.transform(target_value)
    class_names = le.classes_

    categorical_names = {}
    for feature in categorical_features:
        le = sklearn.preprocessing.LabelEncoder()
        le.fit(data_in_arr[:, feature])
        data_in_arr[:, feature] = le.transform(data_in_arr[:, feature])
        categorical_names[feature] = le.classes_
        
    data_in_arr = data_in_arr.astype(float)
    

    encoder = ColumnTransformer([("enc", sklearn.preprocessing.OneHotEncoder(), categorical_features)], remainder = 'passthrough')

    train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(data_in_arr, target_value, random_state=1, train_size=0.80)

    encoder.fit(data_in_arr)
    encoded_train = encoder.transform(train)

    rf_trained_model_file = os.path.join('.', 'rf.pickle')

    if os.path.exists(rf_trained_model_file):
        with open(rf_trained_model_file, 'rb') as f:
            load(f)
    else:
        print('training the rf')
        rf = sklearn.ensemble.RandomForestRegressor(n_estimators=100)
        rf.fit(encoded_train, labels_train)
        dump(rf, rf_trained_model_file)