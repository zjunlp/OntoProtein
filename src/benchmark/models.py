from tape.models.modeling_utils import PairwiseContactPredictionHead
from torch import nn, pdist
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
from torch.nn.utils.weight_norm import weight_norm
from torch.utils.data import Dataset
from transformers import BertPreTrainedModel, BertModel, AdamW, \
    get_linear_schedule_with_warmup

import numpy as np
import torch
from transformers.modeling_outputs import SequenceClassifierOutput, TokenClassifierOutput


class SimpleMLP(nn.Module):

    def __init__(self,
                 in_dim: int,
                 hid_dim: int,
                 out_dim: int,
                 dropout: float = 0.):
        super().__init__()
        self.main = nn.Sequential(
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None))

    def forward(self, x):
        return self.main(x)


class BertForOntoProteinContactPrediction(BertPreTrainedModel):
    def __init__(self, config, mean_output):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)

        self.predict = PairwiseContactPredictionHead(config.hidden_size, ignore_index=-1)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.mean_output = mean_output
        self.init_weights()

    def forward(self, input_ids, protein_length, attention_mask=None, labels=None):
        targets = labels
        outputs = self.bert(input_ids)
        # targets

        sequence_output = outputs[0]
        # print(sequence_output.shape)
        output_precition = self.predict(sequence_output, protein_length, targets) + outputs[2:]
        # (loss), prediction_scores, (hidden_states), (attentions)
        outputs['loss'] = output_precition[0][0]
        outputs['logits'] = output_precition[1]
        outputs['prediction_score'] = output_precition[0][1]
        return outputs


class BertForSequenceClassification2(BertPreTrainedModel):
    def __init__(self, config, mean_output):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        # self.classifier = SimpleMLP(config.hidden_size, 512, config.num_labels)

        self.mean_output = mean_output
        # print(self.mean_output)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if self.mean_output is not True:
            outputs_ = outputs[1]
        else:
            outputs_ = outputs
            attention_mask = attention_mask.bool()
            num_batch_size = attention_mask.size(0)
            outputs_ = torch.stack([outputs_.last_hidden_state[i, attention_mask[i, :], :].mean(dim=0) for i in
                                      range(num_batch_size)], dim=0)

        outputs_ = self.dropout(outputs_)
        logits = self.classifier(outputs_)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def load_adam_optimizer_and_scheduler(model, args, train_dataset):
    optimizer = torch.optim.Adam(model.parameters())

    total_steps = len(
        train_dataset) // args.train_batch_size // args.gradient_accumulation_steps * args.num_train_epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    return optimizer, scheduler


class BertForTokenClassification2(BertPreTrainedModel):
    def __init__(self, config, mean_output):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.mean_output = mean_output

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=True,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


model_mapping = {
    'remote_homology': BertForSequenceClassification2,
    'contact': BertForOntoProteinContactPrediction,
    'fluorescence': BertForSequenceClassification2,
    'stability': BertForSequenceClassification2,
    'ss3': BertForTokenClassification2,
    'ss8': BertForTokenClassification2
}
