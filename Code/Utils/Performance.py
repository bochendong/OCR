import numpy as np
from datasets import load_metric


def compute_metrics(outputs, target, id2label):
    predictions = np.argmax(outputs.logits.detach().cpu().numpy(), axis=2)[0]
    true_predictions = []
    true_labels = []

    counter = 0

    metric = load_metric("seqeval")

    for prediction, label in zip(predictions, target):
        current_prediction = []
        if label != -100:
            counter += 1
            current_prediction.append(id2label[prediction])
        true_predictions.append(current_prediction)

    for prediction, label in zip(predictions, target):
        current_labels = []
        if label != -100:
            current_labels.append(id2label[label.item()])
        true_labels.append(current_labels)

    if counter == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "accuracy": 0.0,
        }
    else:
        results = metric.compute(predictions=true_predictions, references=true_labels, zero_division=0)
        return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }