import os
import json
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import dataclasses
from dataclasses import dataclass
import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer
from transformers import PreTrainedTokenizerBase


@dataclass
class ProteinSeqInputFeatures:
    """
    A single set of feature of data for protein sequences.
    """
    input_ids: List[int]
    label: Optional[Union[int, float]] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"


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


class ProteinSeqDataset(Dataset):
    """
    Dataset for Protein sequence.

    Args:
        data_dir: the diractory need contain pre-train datasets.
        seq_data_file_name: path of sequence data, in view of the multiple corpus choices (e.g. Swiss, UniRef50...), 
                            and only support LMDB file.
        tokenizer: tokenizer used for encoding sequence.
        in_memory: Whether or not to save full sequence data to memory. Suggest that set to `False` 
                   when using UniRef50 or larger corpus.
    """

    def __init__(
        self,
        file_path: str,
        seq_data_path: str = None,
        tokenizer: PreTrainedTokenizerBase = None,
        in_memory: bool=True,
        max_protein_seq_length: int = None
    ):
        self.file_path = file_path
        self.seq_data_path = seq_data_path

        self.protein_seq = [line.rstrip('\n').split('\t')[1] for line in open(file_path, 'r')]

        self.tokenizer = tokenizer
        self.max_protein_seq_length = max_protein_seq_length
        
    def __getitem__(self, index):
        item = self.protein_seq[index]

        # implement padding of sequences at 'DataCollatorForLanguageModeling'
        item = list(item)
        if self.max_protein_seq_length is not None:
            item = item[:self.max_protein_seq_length]
        input_ids = self.tokenizer.encode(item)
        return ProteinSeqInputFeatures(
            input_ids=input_ids,
        )
        
    def __len__(self):
        # return self.num_examples
        return len(self.protein_seq)


class Seq2Vec(nn.Module):
    def __init__(
        self,
        pretrained_model_path: str,
    ):
        super().__init__()

        onto_protein_model = BertModel.from_pretrained(pretrained_model_path)
        self.encoder = onto_protein_model

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=return_dict,
        )
        protein_attention_mask = attention_mask.bool()
        num_batch_size = attention_mask.size(0)
        protein_embedding = torch.stack([outputs.last_hidden_state[i, protein_attention_mask[i, :], :][1:-1].mean(dim=0) for i in range(num_batch_size)], dim=0)
        return protein_embedding

def convert_protein_seq_to_embedding(
    file_path: str,
    pretrained_model_path: str,
    embedding_save_path: str
):

    tokenizer = BertTokenizer.from_pretrained(pretrained_model_path)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, are_protein_length_same=False)

    # Note: default set protein length to 1024.
    protein_seq_dataset = ProteinSeqDataset(
        file_path=file_path,
        tokenizer=tokenizer,
        max_protein_seq_length=1024
    )

    protein_seq_dataloader = DataLoader(
        dataset=protein_seq_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        collate_fn=data_collator
    )

    model = Seq2Vec(pretrained_model_path=pretrained_model_path)
    model.to('cuda:6')

    def to_device(inputs: Dict[str, torch.Tensor]):
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to('cuda:6')
        return inputs
    
    protein_embeddings = []
    for item in tqdm.tqdm(protein_seq_dataloader):
        _ = item.pop('labels')
        inputs = to_device(item)
        with torch.no_grad():
            protein_embedding = model(**inputs).cpu()
            protein_embeddings.append(protein_embedding)
    
    protein_embeddings = torch.cat(protein_embeddings, dim=0)
    
    np.save(embedding_save_path, protein_embeddings)


if __name__ == '__main__':
    convert_protein_seq_to_embedding(
        'data/protein.SHS148k.sequences.dictionary.tsv',
        'data/Ontoprotein_downstream/model/ontoprotein',
        'data/protein_embedding_ontoprotein_1024_shs148k.npy'
    )