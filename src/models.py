from math import gamma
import os
import json
import copy
from typing import Optional, Tuple, Union
import torch
import torch.nn.functional as F
from dataclasses import dataclass
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn.modules.sparse import Embedding
from transformers import PreTrainedModel, AutoConfig, PretrainedConfig, BertModel, BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from transformers.file_utils import ModelOutput
from transformers.utils import logging
from transformers.deepspeed import is_deepspeed_zero3_enabled
from deepspeed import DeepSpeedEngine


logger = logging.get_logger(__name__)


ONTO_CONFIG_NAME = "config.json"
PROTEIN_CONFIG_NAME = "config.json"
PROTEIN_MODEL_STATE_DICT_NAME = 'pytorch_model.bin'
ONTO_MODEL_STATE_DICT_NAME = 'pytorch_model.bin'


class OntoConfig:
    def __init__(self, **kwargs):
        self.use_desc = kwargs.pop('use_desc', False)
        self.ke_embedding_size = kwargs.pop('ke_embedding_size', 512)
        self.double_entity_embedding_size = kwargs.pop('double_entity_embedding_size', True)
        self.gamma = kwargs.pop('gamma', 10.0)
        self.num_relations = kwargs.pop('num_relations', None)
        self.num_go_terms = kwargs.pop('num_go_terms', None)
        self.num_proteins = kwargs.pop('num_proteins', None)
        self.protein_encoder_cls = kwargs.pop('protein_encoder_cls', None)
        self.go_encoder_cls = kwargs.pop('go_encoder_cls', None)

        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                logger.error(f"Can't set {key} with value {value} for {self}")
                raise err

    def save_to_json_file(self, save_directory: os.PathLike):
        os.makedirs(save_directory, exist_ok=True)
        output_config_file = os.path.join(save_directory, ONTO_CONFIG_NAME)

        with open(output_config_file, 'w', encoding='utf-8') as writer:
            writer.write(self._to_json_string())

        logger.info(f'Configuration saved in {output_config_file}')

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

    def _to_json_string(self):
        config_dict = self.to_dict()
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_json_file(cls, config_path: os.PathLike):
        onto_config = cls()
        config_path_file = os.path.join(config_path, ONTO_CONFIG_NAME)
        with open(config_path_file, 'r') as read:
            text = read.read()
        config_dict = json.loads(text)

        for key, value in config_dict.items():
            setattr(onto_config, key, value)

        return onto_config


class ProteinConfig:

    protein_model_config = None

    def save_to_json_file(self, save_directory: os.PathLike):
        os.makedirs(save_directory, exist_ok=True)
        self.protein_model_config.save_pretrained(save_directory)

    @classmethod
    def from_json_file(cls, config_path: os.PathLike):
        config = cls()
        config.protein_model_config = AutoConfig.from_pretrained(config_path)
        return config


@dataclass
class MaskedLMOutput(ModelOutput):
    """
    Base class for masked language models outputs.

    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Masked language modeling (MLM) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    pooler_output: Optional[torch.FloatTensor] = None


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, 512)
        self.activation = nn.Tanh()

    def forward(self, hidden_states, attention_mask):
        # attention_mask = attention_mask.bool()
        # num_batch_size = attention_mask.size(0)
        # pooled_output = torch.stack([hidden_states[i, attention_mask[i, :], :].mean(dim=0) for i in range(num_batch_size)], dim=0)
        pooled_output = hidden_states[:, 0]
        pooled_output = self.dense(pooled_output)
        return pooled_output


class BertForMaskedLM(BertPreTrainedModel):

    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        return_mlm=True
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        prediction_scores = None
        pooler_output = None
        if return_mlm:
            prediction_scores = self.cls(sequence_output)
        else:
            pooler_output = self.pooler(outputs.last_hidden_state, attention_mask)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            pooler_output=pooler_output
        )

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        #  add a dummy token
        assert self.config.pad_token_id is not None, "The PAD token should be defined for generation"
        attention_mask = torch.cat([attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1)
        dummy_token = torch.full(
            (effective_batch_size, 1), self.config.pad_token_id, dtype=torch.long, device=input_ids.device
        )
        input_ids = torch.cat([input_ids, dummy_token], dim=1)

        return {"input_ids": input_ids, "attention_mask": attention_mask}


class OntoProteinPreTrainedModel(nn.Module):
    """
    The model implement the knowledge injection into protein language model. The model consists of two sub-model, 
    KE model and protein language model. 
    
    Note: For convenience of measuring the performence on protein-related downstream task, the state dict of model 
    will be saved to two directory: './Onto' and './Protein', also be loaded from the directory.

    Note: The weights of model could be initialized as pretrained weights using `from_pretrained` method.

    Args:
        protein_model_config: the configuration of protein model.
        onto_model_config: the configuration of ontology model.
    """

    def __init__(
        self,
        protein_model_config = None,
        onto_model_config = None
    ):
        super().__init__()

        self.protein_model_config = protein_model_config,
        self.onto_model_config = onto_model_config
        # onto model: go entity encoder and relation embedding
        self.onto_model = OntoModel(config=onto_model_config)

        # protein model: protein sequence encoder and mlm head.
        self.protein_lm = None
        if onto_model_config.protein_encoder_cls == 'bert':
            self.protein_lm = BertForMaskedLM(protein_model_config)

    def forward(
        self,
        protein_inputs: Tuple = None,
        relation_ids: torch.Tensor = None,
        go_tail_inputs: Union[torch.Tensor, Tuple] = None,
        go_head_inputs: Union[torch.Tensor, Tuple] = None,
        optimize_memory: bool = False
    ):
        head_embed = None
        tail_embed = None
        relation_embed = None

        if protein_inputs:
            protein_input_ids, protein_attention_mask, protein_token_type_ids = protein_inputs
            if not optimize_memory:
                # automatically detect whether or not using protein sequence.
                if protein_attention_mask is not None and protein_token_type_ids is not None:
                    outputs = self.protein_lm(
                        input_ids=protein_input_ids,
                        attention_mask=protein_attention_mask,
                        token_type_ids=protein_token_type_ids,
                        return_dict=True,
                        return_mlm=False
                    )
                # protein_attention_mask = protein_attention_mask.bool()
                # num_batch_size = protein_attention_mask.size(0)
                # head_embed = torch.stack([outputs.last_hidden_state[i, protein_attention_mask[i, :], :].mean(dim=0) for i in range(num_batch_size)], dim=0)
                # head_embed = outputs.last_hidden_state.mean(dim=1)
                    head_embed = outputs.pooler_output
                else:
                    head_embed = self.onto_model.protein_encoder(protein_input_ids)
        if go_head_inputs is not None:
            head_embed, _ = self.onto_model(go_inputs=go_head_inputs)
        
        if head_embed is None and not optimize_memory:
            raise ValueError('Input at least one of protein_inputs and go_head_inputs')

        tail_embed, relation_embed = self.onto_model(go_inputs=go_tail_inputs, relation_ids=relation_ids, optimize_memory=optimize_memory)

        return head_embed, relation_embed, tail_embed
            

    def save_pretrained(
        self,
        save_directory: os.PathLike,
        state_dict: Optional[dict] = None,
        save_config: bool = True,
    ):
        protein_save_directory = os.path.join(save_directory, 'protein')
        onto_save_directory = os.path.join(save_directory, 'onto')
        if self.protein_lm:
            self.protein_lm.save_pretrained(protein_save_directory, save_config=save_config)
        self.onto_model.save_pretrained(onto_save_directory, save_config=save_config)

    @classmethod
    def from_pretrained(
        cls, 
        protein_model_path: os.PathLike, 
        onto_model_path: os.PathLike,
        model_args = None,
        training_args = None,
        **kwargs
    ):

        onto_config_path = onto_model_path
        if not os.path.exists(os.path.join(onto_config_path, ONTO_CONFIG_NAME)):
            logger.info("Don't exist OntoModel config. Will create config according to `OntoProteinModelArguments`")

            # Will feed the number of relations and entity.
            num_relations = kwargs.pop('num_relations')
            num_go_terms = kwargs.pop('num_go_terms')
            num_proteins = kwargs.pop('num_proteins', None)

            onto_config = OntoConfig(
                use_desc=training_args.use_desc,
                ke_embedding_size=model_args.ke_embedding_size,
                double_entity_embedding_size=model_args.double_entity_embedding_size,
                num_relations=num_relations,
                num_go_terms=num_go_terms,
                num_proteins=num_proteins,
                go_encoder_cls=model_args.go_encoder_cls,
                protein_encoder_cls=model_args.protein_encoder_cls,
                gamma=training_args.ke_max_score
            )
        else:
            onto_config = OntoConfig.from_json_file(onto_config_path)
        
        protein_config_path = protein_model_path
        protein_config = ProteinConfig.from_json_file(protein_config_path)
        protein_model_config = protein_config.protein_model_config

        # instantiate model
        onto_protein_model = cls(protein_model_config=protein_model_config, onto_model_config=onto_config)

        # TODO: implement that the protein model class could be choosed.
        if onto_config.protein_encoder_cls == 'bert':
            onto_protein_model.protein_lm = BertForMaskedLM.from_pretrained(protein_model_path)
        onto_protein_model.onto_model = OntoModel.from_pretrained(onto_model_path, config=onto_config)
        
        onto_protein_model.eval()

        return onto_protein_model

class OntoModel(nn.Module):
    """
    The model is sub-model of `OntoProteinPretrainedModel`, which consists of Go term entity embedding (or Go description encoder)
    and the relation embedding of `Go-Go` and `Go-Protein` relations.

    Note: The weights of model could be initialized as pretrained weights using `from_pretrained` method.

    Args:
        config: The configuration of OntoModel.
    """

    def __init__(
        self,
        config = None
    ):
        super().__init__()

        self.config = config
        if config.go_encoder_cls == 'embedding':
            if config.double_entity_embedding_size:
                ke_embedding_size = config.ke_embedding_size * 2
            else:
                ke_embedding_size = config.ke_embedding_size
            self.go_encoder = nn.Embedding(config.num_go_terms, ke_embedding_size)
        elif config.go_encoder_cls == 'bert':
            # self.go_encoder = nn.Embedding(config.vocab_size, config.ke_embedding_size, padding_idx=config.pad_token_id)
            self.go_encoder = None
            if config.double_entity_embedding_size:
                ke_embedding_size = config.ke_embedding_size * 2
            else:
                ke_embedding_size = config.ke_embedding_size
            self.go_encoder_dense = nn.Linear(768, ke_embedding_size)

        self.relation_embedding = nn.Embedding(config.num_relations, config.ke_embedding_size)

        self.protein_encoder = None
        # choice of protein_encoder_cls: {`embedding`, `bert`}
        if config.protein_encoder_cls == 'embedding':
            self.protein_encoder = nn.Embedding(config.num_proteins, config.ke_embedding_size)

        self._init_weights()

    def _init_weights(self):
        """Initialize the weights"""
        if self.config.go_encoder_cls == 'bert':
            state_dict = torch.load('data/model_data/PubMedBERT/pytorch_model.bin')
            embedding_weight = state_dict['bert.embeddings.word_embeddings.weight']
            self.go_encoder = nn.Embedding.from_pretrained(embedding_weight)

        self.embedding_range = (self.config.gamma + 2.0) / self.config.ke_embedding_size
        if isinstance(self.go_encoder, nn.Embedding):
            self.go_encoder.weight.data.uniform_(-self.embedding_range, self.embedding_range)
        
        if isinstance(self.relation_embedding, nn.Embedding):
            self.relation_embedding.weight.data.uniform_(-self.embedding_range, self.embedding_range)
        
        if isinstance(self.protein_encoder, nn.Embedding):
            self.protein_encoder.weight.data.uniform_(-self.embedding_range, self.embedding_range)
        

    def forward(
        self,
        go_inputs: Union[torch.Tensor, Tuple],
        relation_ids: torch.Tensor = None,
        optimize_memory: bool = False
    ):
        entity_embed = None
        relation_embed = None

        if isinstance(go_inputs, Tuple):
            go_input_ids, go_attention_mask, go_token_type_ids = go_inputs
            outputs = self.go_encoder(
                go_input_ids,
            )
            go_attention_mask = go_attention_mask.bool()
            num_batch_size = go_attention_mask.size(0)
            entity_embed = torch.stack([outputs[i, go_attention_mask[i, :], :][1:-1].mean(dim=0) for i in range(num_batch_size)], dim=0)
            entity_embed = self.go_encoder_dense(entity_embed)
        else:
            entity_embed = self.go_encoder(go_inputs)
        
        if relation_ids is not None and not optimize_memory:
            relation_embed = self.relation_embedding(relation_ids)

        return entity_embed, relation_embed
        
    def save_pretrained(
        self, 
        save_directory: os.PathLike,
        save_config: bool = True,
    ):
        os.makedirs(save_directory, exist_ok=True)

        model_to_save = unwrap_model(self)
        state_dict = model_to_save.state_dict()

        if save_config:
            self.config.save_to_json_file(save_directory)
        
        output_model_file = os.path.join(save_directory, ONTO_MODEL_STATE_DICT_NAME)
        torch.save(state_dict, output_model_file)

        logger.info(f'OntoModel weights saved in {output_model_file}')

    @classmethod
    def from_pretrained(cls, model_path, config):
        # TODO: deepspeed
        model = cls(config)

        model_file = os.path.join(model_path, ONTO_MODEL_STATE_DICT_NAME)
        # if not exisit state dict file, it will return random initialized parameters.
        if os.path.exists(model_file):
            state_dict = torch.load(model_file, map_location='cpu')
            model = cls._load_state_dict_into_model(model, state_dict=state_dict)
            return model
        else:
            return model
        
    @classmethod
    def _load_state_dict_into_model(cls, model, state_dict):
        metadata = getattr(state_dict, "_metadata", None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata
        error_msgs = []

        def load(module: nn.Module, prefix=""):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            args = (state_dict, prefix, local_metadata, True, [], [], error_msgs)

            if is_deepspeed_zero3_enabled():
                import deepspeed

                # because zero3 puts placeholders in model params, this context
                # manager gathers (unpartitions) the params of the current layer, then loads from
                # the state dict and then re-partitions them again
                with deepspeed.zero.GatheredParameters(list(module.parameters(recurse=False)), modifier_rank=0):
                    if torch.distributed.get_rank() == 0:
                        module._load_from_state_dict(*args)
            else:
                module._load_from_state_dict(*args)

            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + ".")
        
        start_prefix = ""
        load(model, start_prefix)
        
        return model


@dataclass
class OntoProteinMLMLoss:
    """
    Loss function for MLM.

    Args:
        mlm_lambda: hyper-parameters to control the effect of MLM loss.
    """
    def __init__(self, mlm_lambda=1.0):
        self.mlm_lambda = mlm_lambda

    def __call__(
        self,
        model: OntoProteinPreTrainedModel,
        **kwargs
    ):
        protein_mlm_input_ids = kwargs.pop('input_ids', None)
        protein_mlm_attention_mask = kwargs.pop('attention_mask', None)
        protein_mlm_token_type_ids = kwargs.pop('token_type_ids', None)
        protein_mlm_labels = kwargs.pop('labels', None)

        if isinstance(model, DeepSpeedEngine):
            mlm_outputs = model.module.protein_lm(
                input_ids=protein_mlm_input_ids,
                attention_mask=protein_mlm_attention_mask,
                token_type_ids=protein_mlm_token_type_ids,
                labels=protein_mlm_labels,
                inputs_embeds=None
            )
        else:
            mlm_outputs = model.protein_lm(
                input_ids=protein_mlm_input_ids,
                attention_mask=protein_mlm_attention_mask,
                token_type_ids=protein_mlm_token_type_ids,
                labels=protein_mlm_labels,
                inputs_embeds=None
            )
        mlm_loss = mlm_outputs.loss

        return mlm_loss * self.mlm_lambda

class OntoProteinKELoss:
    """
    Loss function for KE.

    Args:
        ke_lambda: hyper-parameter to control the effect of KE loss.
        max_score: maximal score for triplet.
        num_neg_sample: the number of negative sampling, which will be used to normalize KE loss of negative samples.
    """
    def __init__(self, ke_lambda=1.0, max_score=None, score_fn='transE', num_protein_go_neg_sample=1, num_go_go_neg_sample=1):
        self.ke_lambda = ke_lambda
        self.max_score = max_score
        self.num_protein_go_neg_sample = num_protein_go_neg_sample
        self.num_go_go_neg_sample = num_go_go_neg_sample
        self.score_fn = score_fn

    def __call__(
        self,
        model: OntoProteinPreTrainedModel,
        triplet_type: str = 'protein-go',
        is_neg: bool = False,
        use_desc: bool = False,
        use_seq: bool = True,
        optimize_memory: bool = True,
        **kwargs
    ):
        head_input_ids = kwargs.pop('head_input_ids')
        head_attention_mask = kwargs.pop('head_attention_mask', None)
        head_token_type_ids = kwargs.pop('head_token_type_ids', None)

        relation_ids = kwargs.pop('relation_ids')

        head_relation_embed = kwargs.pop('cache_head_relation_embed', None)

        tail_input_ids = kwargs.pop('tail_input_ids')
        tail_attention_mask = kwargs.pop('tail_attention_mask', None)
        tail_token_type_ids = kwargs.pop('tail_token_type_ids', None)

        go_tail_inputs = tail_input_ids
        if use_desc:
            go_tail_inputs = (tail_input_ids, tail_attention_mask, tail_token_type_ids)

        if triplet_type == 'protein-go':
            # if set `optimize_memory` to `True`, `head_embed` and `relation_embed` will be `None`
            head_embed, relation_embed, tail_embed = model(
                protein_inputs=(head_input_ids, head_attention_mask, head_token_type_ids),
                relation_ids=relation_ids,
                go_tail_inputs = go_tail_inputs,
                optimize_memory=optimize_memory
            )
        elif triplet_type == 'go-go':
            go_head_inputs = head_input_ids
            if use_desc:
                go_head_inputs = (head_input_ids, head_attention_mask, head_token_type_ids)

            head_embed, relation_embed, tail_embed = model(
                go_head_inputs=go_head_inputs,
                relation_ids=relation_ids,
                go_tail_inputs=go_tail_inputs,
            )
        else:
            raise Exception(f'Not support {triplet_type} triplet type.')

        embedding_range = model.module.onto_model.embedding_range
        if self.score_fn == 'transE':
            score, head_relation_embed = self._transe_score(tail_embed, head_embed, relation_embed, head_relation_embed)
        elif self.score_fn == 'rotatE':
            score, head_relation_embed = self._rotate_score(tail_embed, embedding_range, head_embed, relation_embed, head_relation_embed)
        else:
            raise ValueError("invalid score function.")

        if is_neg:
            ke_loss = -1.0 * F.logsigmoid(score - self.max_score).mean()
        else:
            ke_loss = -1.0 * F.logsigmoid(self.max_score - score).mean()
        return ke_loss * self.ke_lambda / 2, head_relation_embed

    def _transe_score(self, tail_embed, head_embed=None, relation_embed=None, head_relation_embed=None):
        
        if head_relation_embed is None:
            head_relation_embed = head_embed + relation_embed
        else:
            num_pos = head_relation_embed.size(0)
            num_neg = tail_embed.size(0)

            if num_pos != num_neg:
                assert num_neg % num_pos == 0
                tail_embed = tail_embed.view(num_pos, num_neg // num_pos, -1).permute(1, 0, 2)
        
        score = (- tail_embed + head_relation_embed).norm(p=1, dim=-1)

        return score, head_relation_embed

    def _rotate_score(self, tail_embed, embedding_range, head_embed=None, relation_embed=None, head_relation_embed=None):
        pi = 3.14159265358979323846

        tail_re, tail_im = tail_embed.chunk(2, dim=-1)

        if head_relation_embed is None:
            head_re, head_im = head_embed.chunk(2, dim=-1)

            relation_phase = relation_embed / (embedding_range / pi)
            relation_re, relation_im = torch.cos(relation_phase), torch.sin(relation_phase)
        
            head_relation_re = head_re * relation_re - head_im * relation_im
            head_relation_im = head_re * relation_im - head_im * relation_re
        else:
            head_relation_re, head_relation_im = head_relation_embed
            num_pos = head_relation_re.size(0)
            num_neg = tail_embed.size(0)

            if num_pos != num_neg:
                assert num_neg % num_pos == 0
                tail_re = tail_re.view(num_pos, num_neg // num_pos, -1).permute(1, 0, 2)
                tail_im = tail_im.view(num_pos, num_neg // num_pos, -1).permute(1, 0, 2)

        score_re =  head_relation_re - tail_re
        score_im =  head_relation_im - tail_im

        score = torch.stack([score_re, score_im], dim=0).norm(dim=0)
        score = score.sum(dim=-1)
        
        return score, (head_relation_re, head_relation_im)

                
def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Recursively unwraps a model from potential containers (as used in distributed training).

    Args:
        model (:obj:`torch.nn.Module`): The model to unwrap.
    """
    # since there could be multiple levels of wrapping, unwrap recursively
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model
