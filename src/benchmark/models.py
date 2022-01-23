import torch.nn as nn
from transformers import AutoModelForTokenClassification, BertPreTrainedModel, AutoModel


model_fn_mapping = {
    'ssp': AutoModelForTokenClassification,
}