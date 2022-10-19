import pandas as pd
import numpy as np


def load_data():
    tests_complete = pd.read_pickle('dataset/test-case-data/full_regression_data_complete.pkl')
    display(tests_complete.head())
    return tests_complete

def get_test_data_info(test_cases):
    
    # Feature coverage info
    feature_coverage_full_suite = {}

    # Dict to store in which test cases each feature appears
    feature_test_id_relation = {}

    # Execution time
    execution_time_full_suite = {}

    # Dict with mapping between integer id and test case key
    mapping_test_key_full_suite = {}

    index_test_case = 0

    for index,row in test_cases.iterrows():
        feature_coverage_full_suite[index_test_case] = row['labels']

        execution_time_full_suite[index_test_case] = row['estimated_time']

        mapping_test_key_full_suite[index_test_case] = row['test_key']

        for feat in row['labels']:
            if feat not in feature_test_id_relation:
                feature_test_id_relation[feat] = [index_test_case]
            else:
                existing_test_ids = feature_test_id_relation[feat]
                existing_test_ids.append(index_test_case)
                feature_test_id_relation[feat] = existing_test_ids

        index_test_case += 1

    total_execution_time_full_suite = sum(execution_time_full_suite.values())
    print("Total execution time: {total_time}".format(total_time=total_execution_time_full_suite))
    
    return feature_coverage_full_suite, execution_time_full_suite, mapping_test_key_full_suite, feature_test_id_relation, total_execution_time_full_suite, index_test_case
        

def build_feature_mapping():
    # Build mapping (from Snowflake data to testing data)
    mapping_file = open('PATH_OF_FILE_WITH_FEATURE_NAMES_SNOWFLAKE_TEST_CASES')
    mapping_dict = {}
    
    for line in mapping_file:
        full_line = line.split(':')
        game_data_feature_name = full_line[0].strip()
        testing_feature_name = full_line[1].strip()
        mapping_dict[game_data_feature_name] = testing_feature_name
    return mapping_dict

def remove_features_mapping(mapping_dict, num_uses_features):
    # Check features that are not in the mapping (e.g., corrupted data from game data events)
    all_features = num_uses_features['Feature'].to_list()
    features_remove = []
    for feat in all_features:
        try:
            testing_feat = mapping_dict[feat]
        except:
            features_remove.append(feat)

    print("Features to be removed: ", features_remove)

    # Remove features that are not in the mapping
    num_uses_features = num_uses_features.loc[~num_uses_features.Feature.isin(features_remove)]
    print("Number of features after removing features that are not in the mapping: {num}".format(num=len(num_uses_features)))