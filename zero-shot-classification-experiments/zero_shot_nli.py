# Zero-shot with NLI
import torch
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import os
import string
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import multilabel_confusion_matrix, precision_recall_fscore_support


# Def function to run zero shot nli
def run_zero_shot_nli(zero_shot_nli_classifier, candidate_labels, df, experiment_name, run_name, experiment_dir):

    mlflow.set_tracking_uri(experiment_dir)
    experiment_active = mlflow.set_experiment(experiment_name)

    # Instantiate multi-label binarizer (one-hot encoded labels) and apply it to our labels
    mlb  = MultiLabelBinarizer()
    mlb.fit([candidate_labels])

    print("Running for " + run_name + "...")

    # Create dataframe to store predictions
    prodigy_test_case_data_predictions = pd.DataFrame(columns=['id', 'type', 'description', 'threshold', 'labels', 'predicted_labels',
                                                               'label_encoded', 'predicted_label_encoded', 'prediction_prob_score'])
    index_add = 0

    # Thresholds to include predicted labels (prediction confidence - probability)
    threshold_pred = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    for index,row in df.iterrows():
        sequence_classify  = row['description']
        classification_res = zero_shot_nli_classifier(sequence_classify, candidate_labels, multi_label=True)
        predicted_labels = classification_res['labels']
        prediction_scores = classification_res['scores']

        # Get prediction for test case name
        for threshold in threshold_pred:

            # Iterate through predictions (descendent order by score)
            labels_to_include = set()
            scores_to_include = []
            for label,score in zip(predicted_labels, prediction_scores):
                if score < threshold:
                    break
                else:
                    labels_to_include.add(label)
                    scores_to_include.append(score)

            # Encode labels: ground truth and predicted
            correct_labels = row['labels']
            encoded_correct_labels = mlb.transform([correct_labels])
            encoded_labels_to_include = mlb.transform([labels_to_include])

            # Update prediction df
            prodigy_test_case_data_predictions.loc[index_add] = [row['id'], row['type'], row['description'], threshold, correct_labels, labels_to_include,
                                                                 encoded_correct_labels, encoded_labels_to_include, scores_to_include]
            index_add += 1           

    # Get df for each threshold
    for threshold in threshold_pred:
        with mlflow.start_run(experiment_id=experiment_active.experiment_id, run_name=run_name):
            
            threshold_df = prodigy_test_case_data_predictions[prodigy_test_case_data_predictions['threshold'] == threshold]
            
            y_true = []
            y_pred = []
            for index,row in threshold_df.iterrows():
                label_encoded = row['label_encoded']
                predicted_label_encoded = row['predicted_label_encoded']
                y_true.append(label_encoded[0])
                y_pred.append(predicted_label_encoded[0])

            # Get metrics
            metrics = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
            precision = metrics[0]
            recall = metrics[1]
            fscore = metrics[2]
            mlflow.log_param("confidence_threshold", threshold)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("fscore", fscore)

    print("Execution for " + run_name + " finished!")