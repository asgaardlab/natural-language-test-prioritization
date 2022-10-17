import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import os
import string
import time
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import multilabel_confusion_matrix, precision_recall_fscore_support


def run_baseline(df, candidate_labels, candidate_labels_mod, experiment_name, run_name, experiment_dir):

    mlflow.set_tracking_uri(experiment_dir)
    experiment_active = mlflow.set_experiment(experiment_name)

    # Instantiate multi-label binarizer (one-hot encoded labels) and apply it to our labels
    mlb  = MultiLabelBinarizer()
    mlb.fit([candidate_labels])

    print("Running for " + run_name + "...")

    # Create dataframe to store predictions
    prodigy_test_case_data_predictions = pd.DataFrame(columns=['id', 'type', 'description', 'labels', 'predicted_labels',
                                                               'label_encoded', 'predicted_label_encoded'])
    index_add = 0

    start_time = time.time()

    # Iterate through labeled data dataframe
    for index,row in df.iterrows():
        sequence_classify  = row['description']
        labels_to_include = set()
        scores_to_include = []

        # Check if label is in the test case textual description
        for label, label_mod in zip(candidate_labels, candidate_labels_mod):
            if label_mod in sequence_classify.lower():
                labels_to_include.add(label)
        
        # Encode labels: ground truth and predicted
        correct_labels = row['labels']
        encoded_correct_labels = mlb.transform([correct_labels])
        encoded_labels_to_include = mlb.transform([labels_to_include])

        # Update df with predictions
        prodigy_test_case_data_predictions.loc[index_add] = [row['id'], row['type'], row['description'], correct_labels, labels_to_include,
                                                             encoded_correct_labels, encoded_labels_to_include]
        index_add += 1           

    with mlflow.start_run(experiment_id=experiment_active.experiment_id, run_name=run_name):
        y_true = []
        y_pred = []
        for index,row in prodigy_test_case_data_predictions.iterrows():
            label_encoded = row['label_encoded']
            predicted_label_encoded = row['predicted_label_encoded']
            y_true.append(label_encoded[0])
            y_pred.append(predicted_label_encoded[0])
            
        # Get metrics
        metrics = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
        precision = metrics[0]
        recall = metrics[1]
        fscore = metrics[2]
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("fscore", fscore)   

    print("Execution for " + run_name + " finished!")
    end_time = time.time()
    print("Execution finished with " + str((end_time - start_time)/60) + " minutes.")