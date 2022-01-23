import torch
from dataclasses import dataclass
from torch._C import dtype
from transformers import PreTrainedTokenizerBase
from typing import List, Dict, Optional, Tuple

from src.dataset import ProteinGoInputFeatures, GoGoInputFeatures, ProteinSeqInputFeatures


def _collate_batch_for_protein_seq(
    examples: List[Dict], 
    tokenizer: PreTrainedTokenizerBase,
    are_protein_length_same: bool
):
    if isinstance(examples[0], ProteinSeqInputFeatures):
        examples = [torch.tensor(e.input_ids, dtype=torch.long) for e in examples]

    if are_protein_length_same:
        return torch.stack(examples, dim=0)

    max_length = max(x.size(0) for x in examples)
    result = examples[0].new_full([len(examples), max_length], fill_value=tokenizer.pad_token_id)
    for i, example in enumerate(examples):
        if tokenizer.padding_side == 'right':
            result[i, :example.size(0)] = example
        else:
            result[i, -example.size(0):] = example
    return result


def _collate_batch_for_protein_go(
    examples: List[ProteinGoInputFeatures],
    protein_tokenizer: PreTrainedTokenizerBase,
    text_tokenizer: PreTrainedTokenizerBase,
    are_protein_length_same: bool
):
    assert isinstance(examples[0], ProteinGoInputFeatures), "Only support `ProteinGoInputFeatures`"

    use_seq = False
    use_desc = False
    if not isinstance(examples[0].postive_go_input_ids, int):
        use_desc = True
    
    if not isinstance(examples[0].postive_protein_input_ids, int):
        use_seq = True
    
    # collate postive samples
    # protein
    if use_seq:
        # use sequence
        all_postive_protein_input_ids = [torch.tensor(example.postive_protein_input_ids, dtype=torch.long) for example in examples]
        if are_protein_length_same:
            all_postive_protein_input_ids = torch.stack(all_postive_protein_input_ids, dim=0)
        else:
            max_length = max(x.size(0) for x in all_postive_protein_input_ids)
            all_postive_protein_input_ids_ = all_postive_protein_input_ids[0].new_full([len(all_postive_protein_input_ids), max_length], fill_value=protein_tokenizer.pad_token_id)
            for i, postive_protein_input_ids in enumerate(all_postive_protein_input_ids):
                if protein_tokenizer.padding_side == 'right':
                    all_postive_protein_input_ids_[i, :postive_protein_input_ids.size(0)] = postive_protein_input_ids
                else:
                    all_postive_protein_input_ids_[i, -postive_protein_input_ids.size(0):] = postive_protein_input_ids
            all_postive_protein_input_ids = all_postive_protein_input_ids_
    else:
        # not use sequence
        all_postive_protein_input_ids = torch.tensor([example.postive_protein_input_ids for example in examples], dtype=torch.long)
    # relation
    all_postive_relation_ids = torch.tensor([example.postive_relation_ids for example in examples], dtype=torch.long)
    # go term
    if use_desc:
        all_postive_go_input_ids = [torch.tensor(example.postive_go_input_ids, dtype=torch.long) for example in examples]
        all_postive_go_input_ids = torch.stack(all_postive_go_input_ids, dim=0)
    else:
        all_postive_go_input_ids = torch.tensor([example.postive_go_input_ids for example in examples], dtype=torch.long)
    
    # collate negative samples
    # protein
    all_negative_protein_input_ids = None
    # if use_seq:
    #     for example in examples:
    #         all_negative_protein_input_ids.extend([torch.tensor(negative_protein_input_ids, dtype=torch.long) for negative_protein_input_ids in example.negative_protein_input_ids])
    #     length_of_first_protein = all_negative_protein_input_ids[0].size(0)
    #     are_protein_tensors_same = all(x.size(0) == length_of_first_protein for x in all_negative_protein_input_ids)
    #     if are_protein_tensors_same:
    #         all_negative_protein_input_ids = torch.stack(all_negative_protein_input_ids, dim=0)
    #     else:
    #         max_length = max(x.size(0) for x in all_negative_protein_input_ids)
    #         all_negative_protein_input_ids_ = all_negative_protein_input_ids[0].new_full([len(all_negative_protein_input_ids), max_length], fill_value=protein_tokenizer.pad_token_id)
    #         for i, negative_protein_input_ids in enumerate(all_negative_protein_input_ids):
    #             if protein_tokenizer.padding_side == 'right':
    #                 all_negative_protein_input_ids_[i, :negative_protein_input_ids.size(0)] = negative_protein_input_ids
    #             else:
    #                 all_negative_protein_input_ids_[i: -negative_protein_input_ids.size(0)] = negative_protein_input_ids
    #         all_negative_protein_input_ids = all_negative_protein_input_ids_
    # else:
    #     for example in examples:
    #         all_negative_protein_input_ids.extend(example.negative_protein_input_ids)
    #     all_negative_protein_input_ids = torch.tensor(all_negative_protein_input_ids, dtype=torch.long)
    # relation
    all_negative_relation_ids = []
    for example in examples:
        all_negative_relation_ids.extend(example.negative_relation_ids)
    all_negative_relation_ids = torch.tensor(all_negative_relation_ids, dtype=torch.long)
    # go term
    all_negative_go_input_ids = []
    for example in examples:
        all_negative_go_input_ids.extend(example.negative_go_input_ids)
    all_negative_go_input_ids = torch.tensor(all_negative_go_input_ids, dtype=torch.long)

    all_postive_protein_attention_mask = None
    all_postive_protein_token_type_ids = None
    all_negative_protein_attention_mask = None
    all_negative_protein_token_type_ids = None
    if use_seq:
        all_postive_protein_attention_mask = (all_postive_protein_input_ids != protein_tokenizer.pad_token_id).long()
        all_postive_protein_token_type_ids = torch.zeros_like(all_postive_protein_input_ids, dtype=torch.long)
        # all_negative_protein_attention_mask = (all_negative_protein_input_ids != protein_tokenizer.pad_token_id).long()
        # all_negative_protein_token_type_ids = torch.zeros_like(all_negative_protein_input_ids, dtype=torch.long)

    all_postive_go_attention_mask = None
    all_postive_go_token_type_ids = None
    all_negative_go_attention_mask = None
    all_negative_go_token_type_ids = None
    if use_desc:
        all_postive_go_attention_mask = (all_postive_go_input_ids != text_tokenizer.pad_token_id).long()
        all_postive_go_token_type_ids = torch.zeros_like(all_postive_go_input_ids, dtype=torch.long)
        all_negative_go_attention_mask = (all_negative_go_input_ids != text_tokenizer.pad_token_id).long()
        all_negative_go_token_type_ids = torch.zeros_like(all_negative_go_input_ids, dtype=torch.long)

    return {
        'postive': {
            'head_input_ids': all_postive_protein_input_ids,
            'head_attention_mask': all_postive_protein_attention_mask,
            'head_token_type_ids': all_postive_protein_token_type_ids,
            'relation_ids': all_postive_relation_ids,
            'tail_input_ids': all_postive_go_input_ids,
            'tail_attention_mask': all_postive_go_attention_mask,
            'tail_token_type_ids': all_postive_go_token_type_ids
        },
        'negative': {
            'head_input_ids': all_negative_protein_input_ids,
            'head_attention_mask': all_negative_protein_attention_mask,
            'head_token_type_ids': all_negative_protein_token_type_ids,
            'relation_ids': all_negative_relation_ids,
            'tail_input_ids': all_negative_go_input_ids,
            'tail_attention_mask': all_negative_go_attention_mask,
            'tail_token_type_ids': all_negative_go_token_type_ids
        }
    }


def _collate_batch_for_go_go(
    examples: List[GoGoInputFeatures],
    tokenizer: PreTrainedTokenizerBase,
):
    assert isinstance(examples[0], GoGoInputFeatures), "Only support `GoGoInputFeatures`"
    
    use_desc = False
    if not isinstance(examples[0].postive_go_head_input_ids, int):
        use_desc = True
    #collate postive samples.
    # Go head
    if use_desc:
        all_postive_go_head_input_ids = [torch.tensor(example.postive_go_head_input_ids, dtype=torch.long) for example in examples]
        all_postive_go_head_input_ids = torch.stack(all_postive_go_head_input_ids, dim=0)
    else:
        all_postive_go_head_input_ids = torch.tensor([example.postive_go_head_input_ids for example in examples], dtype=torch.long)
    # relation
    all_postive_relation_ids = torch.tensor([example.postive_relation_ids for example in examples], dtype=torch.long)
    # Go tail
    if use_desc:
        all_postive_go_tail_input_ids = [torch.tensor(example.postive_go_tail_input_ids, dtype=torch.long) for example in examples]
        all_postive_go_tail_input_ids = torch.stack(all_postive_go_tail_input_ids, dim=0)
    else:
        all_postive_go_tail_input_ids = torch.tensor([example.postive_go_tail_input_ids for example in examples], dtype=torch.long)

    # collate negative samples.
    # Go head
    all_negative_go_head_input_ids = []
    for example in examples:
        all_negative_go_head_input_ids.extend(example.negative_go_head_input_ids)
    all_negative_go_head_input_ids = torch.tensor(all_negative_go_head_input_ids, dtype=torch.long)
    # relation
    all_negative_relation_ids = []
    for example in examples:
        all_negative_relation_ids.extend(example.negative_relation_ids)
    all_negative_relation_ids = torch.tensor(all_negative_relation_ids, dtype=torch.long)
    # Go tail
    all_negative_go_tail_input_ids = []
    for example in examples:
        all_negative_go_tail_input_ids.extend(example.negative_go_tail_input_ids)
    all_negative_go_tail_input_ids = torch.tensor(all_negative_go_tail_input_ids, dtype=torch.long)

    all_postive_go_head_attention_mask = None
    all_postive_go_head_token_type_ids = None
    all_postive_go_tail_attention_mask = None
    all_postive_go_tail_token_type_ids = None
    all_negative_go_head_attention_mask = None
    all_negative_go_head_token_type_ids = None
    all_negative_go_tail_attention_mask = None
    all_negative_go_tail_token_type_ids = None
    if use_desc:
        all_postive_go_head_attention_mask = (all_postive_go_head_input_ids != tokenizer.pad_token_id).long()
        all_postive_go_head_token_type_ids = torch.zeros_like(all_postive_go_head_input_ids, dtype=torch.long)
        all_negative_go_head_attention_mask = (all_negative_go_head_input_ids != tokenizer.pad_token_id).long()
        all_negative_go_head_token_type_ids = torch.zeros_like(all_negative_go_head_input_ids, dtype=torch.long)
        all_postive_go_tail_attention_mask = (all_postive_go_tail_input_ids != tokenizer.pad_token_id).long()
        all_postive_go_tail_token_type_ids = torch.zeros_like(all_postive_go_tail_input_ids, dtype=torch.long)
        all_negative_go_tail_attention_mask = (all_negative_go_tail_input_ids != tokenizer.pad_token_id).long()
        all_negative_go_tail_token_type_ids = torch.zeros_like(all_negative_go_tail_input_ids, dtype=torch.long)

    return {
        'postive': {
            'head_input_ids': all_postive_go_head_input_ids,
            'head_attention_mask': all_postive_go_head_attention_mask,
            'head_token_type_ids': all_postive_go_head_token_type_ids,
            'relation_ids': all_postive_relation_ids,
            'tail_input_ids': all_postive_go_tail_input_ids,
            'tail_attention_mask': all_postive_go_tail_attention_mask,
            'tail_token_type_ids': all_postive_go_tail_token_type_ids
        },
        'negative': {
            'head_input_ids': all_negative_go_head_input_ids,
            'head_attention_mask': all_negative_go_head_attention_mask,
            'head_token_type_ids': all_negative_go_head_token_type_ids,
            'relation_ids': all_negative_relation_ids,
            'tail_input_ids': all_negative_go_tail_input_ids,
            'tail_attention_mask': all_negative_go_tail_attention_mask,
            'tail_token_type_ids': all_negative_go_tail_token_type_ids
        }
    }


@dataclass
class DataCollatorForGoGo:
    """
    Data collator used for KE model which the type of dataset is `GoGoDataset`
    """
    tokenizer: PreTrainedTokenizerBase

    def __call__(
        self,
        examples: List[GoGoInputFeatures]
    ) -> Dict[str, torch.Tensor]:
        batch = _collate_batch_for_go_go(examples, self.tokenizer)
        return batch


@dataclass
class DataCollatorForProteinGo:
    """
    Data collator used for KE model which the type of dataset is `ProteinGoDataset`

    Args:
        protein_tokenizer: the tokenizer used for encoding protein sequence.
        are_protein_length_same: If the length of proteins in a batch is different, protein sequence will
                                 are dynamically padded to the maximum length in a batch.
    """

    protein_tokenizer: PreTrainedTokenizerBase
    text_tokenizer: PreTrainedTokenizerBase
    are_protein_length_same: bool = False

    def __call__(
        self,
        examples: List[ProteinGoInputFeatures]
    ) -> Dict[str, torch.Tensor]:
        batch = _collate_batch_for_protein_go(examples, self.protein_tokenizer, self.text_tokenizer, self.are_protein_length_same)
        return batch


@dataclass
class DataCollatorForLanguageModeling:
    """
    Data collator used for language model. Inputs are dynamically padded to the maximum length
    of a batch if they are not all of the same length.
    The class is rewrited from 'Transformers.data.data_collator.DataCollatorForLanguageModeling'.
        
    Agrs:
        tokenizer: the tokenizer used for encoding sequence.
        mlm: Whether or not to use masked language modeling. If set to 'False', the labels are the same as the
            inputs with the padding tokens ignored (by setting them to -100). Otherwise, the labels are -100 for
            non-masked tokens and the value to predict for the masked token.
        mlm_probability: the probablity of masking tokens in a sequence.
        are_protein_length_same: If the length of proteins in a batch is different, protein sequence will
                                 are dynamically padded to the maximum length in a batch.
    """

    tokenizer: PreTrainedTokenizerBase
    mlm: bool = True
    mlm_probability: float = 0.15
    are_protein_length_same: bool = False

    def __post_init__(self):
        if self.mlm and self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead."
            )
    
    def __call__(
        self,
        examples: List[Dict],
    ) -> Dict[str, torch.Tensor]:
        batch = {'input_ids': _collate_batch_for_protein_seq(examples, self.tokenizer, self.are_protein_length_same)}
        special_tokens_mask = batch.pop('special_tokens_mask', None)
        if self.mlm:
            batch['input_ids'], batch['labels'] = self.mask_tokens(
                batch['input_ids'], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch['input_ids'].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch['labels'] = labels

        batch['attention_mask'] = (batch['input_ids'] != self.tokenizer.pad_token_id).long()
        batch['token_type_ids'] = torch.zeros_like(batch['input_ids'], dtype=torch.long)
        return batch

    def mask_tokens(
        self,
        inputs: torch.Tensor,
        special_tokens_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling:
        default: 80% MASK, 10%  random, 10% original
        """
        labels = inputs.clone()
        probability_matrix = torch.full(labels.size(), fill_value=self.mlm_probability)
        # if `special_tokens_mask` is None, generate it by `labels`
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        # only compute loss on masked tokens.
        labels[~masked_indices] = -100

        # 80% of the time, replace masked input tokens with tokenizer.mask_token
        indices_replaced = torch.bernoulli(torch.full(labels.shape, fill_value=0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, fill_value=0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        return inputs, labels
