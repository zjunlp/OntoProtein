import numpy as np
import torch
from typing import Callable, Dict
from transformers.trainer_utils import EvalPrediction
from transformers.trainer_pt_utils import LabelSmoother


def simple_accuracy(preds, labels):
    return (preds == labels).float().mean()


def bt_compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels), f"Predictions and labels have mismatched lengths {len(preds)} and {len(labels)}"

    # TODO: complement remain tasks' metrics
    if task_name == 'ssp':
        return {'acc': simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)


def build_compute_metrics_fn(task_name: str, output_type: str) -> Callable[[EvalPrediction], Dict]:
    def compute_metrics_fn(p: EvalPrediction):
        if output_type == 'token-level-classification':
            logits = p.predictions
            preds = np.argmax(logits, axis=-1)
            label_ids = torch.from_numpy(p.label_ids)
            preds = torch.from_numpy(preds)

            active_index = (label_ids.view(-1) != -100)
            active_preds = preds.view(-1)[active_index]
            active_labels = label_ids.view(-1)[active_index]
            return compute_metrics_mapping[task_name](task_name, active_preds, active_labels)
        elif output_type == 'seq-level-classification':
            logits = p.predictions
            preds = np.argmax(logits, axis=1)
            label_ids = p.label_ids
            return compute_metrics_mapping[task_name](task_name, preds, label_ids)
        else:
            raise Exception("output type not supported.")
    
    return compute_metrics_fn


compute_metrics_mapping = {
    'ssp': bt_compute_metrics,

}