from nltk.tokenize import word_tokenize, TweetTokenizer
import nltk 
from nltk.corpus import stopwords
import pathlib
from pathlib import Path
import pandas as pd
import numpy as np
import os
import time
import string

# Instantiante tokenizer
tokenizer = TweetTokenizer()


def read_data():

    # Load labeled data
    cwd = pathlib.Path().resolve()
    labeled_data_dir = "INSERT_DIR_OF_TEST_CASE_DATA"
    dataset_dir = os.path.join(str(Path(cwd).parents[1]), labeled_data_dir)
    labeled_test_cases_df = pd.read_pickle(dataset_dir)
    return labeled_test_cases_df


def preprocess_data(df):
    prodigy_testing_data = pd.DataFrame(columns=['id', 'type', 'description', 'labels'])
    index_add = 0

    for index,row in df.iterrows():
        # Get test case name
        df_id = row['key']
        df_type = 'test case name'
        df_description = row['name']

        list_labels = row[['my_label_1', 'my_label_2', 'my_label_3', 'my_label_4']].to_list()
        df_labels = set([x for x in list_labels if not pd.isnull(x)])

        prodigy_testing_data.loc[index_add] = [df_id, df_type, df_description, df_labels]
        index_add += 1

        # Get test case name + objective
        df_id = row['key']
        df_type = 'test case name and objective'
        df_description = row['name'] + ". " + row['objective']

        list_labels = row[['my_label_1', 'my_label_2', 'my_label_3', 'my_label_4']].to_list()
        df_labels = set([x for x in list_labels if not pd.isnull(x)])

        prodigy_testing_data.loc[index_add] = [df_id, df_type, df_description, df_labels]
        index_add += 1

    # Split test case and test steps into different dataframes
    prodigy_test_case_name = prodigy_testing_data[prodigy_testing_data['type'] == 'test case name']
    prodigy_test_case_name_obj = prodigy_testing_data[prodigy_testing_data['type'] == 'test case name and objective']
    
    print(f"There are {len(prodigy_testing_data)} rows in the dataset.")
    print(f"There are {len(prodigy_test_case_name)} rows in the test case name dataset.")
    print(f"There are {len(prodigy_test_case_name_obj)} rows in the test case name + objective dataset.")

    # Deduplicate dfs
    # Test case name
    total_duplicate_test_cases = sum(prodigy_test_case_name['description'].duplicated())
    print(f"There are {total_duplicate_test_cases} duplicate test cases.")

    prodigy_test_case_name = prodigy_test_case_name[~prodigy_test_case_name["description"].duplicated()]
    print(f"There are {len(prodigy_test_case_name)} rows in the deduplicated test case name dataset.")

    # Test case name + objective
    total_duplicate_test_cases = sum(prodigy_test_case_name_obj['description'].duplicated())
    print(f"There are {total_duplicate_test_cases} duplicate test cases.")

    prodigy_test_case_name_obj = prodigy_test_case_name_obj[~prodigy_test_case_name_obj["description"].duplicated()]
    print(f"There are {len(prodigy_test_case_name_obj)} rows in the deduplicated test case name + obj dataset.")

    return prodigy_test_case_name, prodigy_test_case_name_obj