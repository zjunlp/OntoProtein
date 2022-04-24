import os
from posixpath import join
from sys import path
import time
import lmdb
import torch
import json
import numpy as np
import pickle as pkl
import dataclasses
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


def _split_go_by_type(go_types) -> Dict[str, List]:
    component_go = []
    function_go = []
    process_go = []
    for go_id, type_ in go_types.items():
        if type_ == 'Process':
            process_go.append(go_id)
        elif type_ == 'Component':
            component_go.append(go_id)
        elif type_ == 'Function':
            function_go.append(go_id)
        else:
            raise Exception('the type not supported.')

    go_terms_type_dict = {
        'Process': process_go,
        'Component': component_go,
        'Function': function_go
    }

    return go_terms_type_dict


def get_triplet_data(data_path):
    heads = []
    relations = []
    tails = []
    true_tail = {}
    true_head = {}

    for line in open(data_path, 'r'):
        head, relation, tail = [int(id) for id in line.rstrip('\n').split()]
        heads.append(head)
        relations.append(relation)
        tails.append(tail)

        if (head, relation) not in true_tail:
            true_tail[(head, relation)] = []
        true_tail[(head, relation)].append(tail)
        if (relation, tail) not in true_head:
            true_head[(relation, tail)] = []
        true_head[(relation, tail)].append(head)

    true_tail = {key: np.array(list(set(val))) for key, val in true_tail.items()}
    true_head = {key: np.array(list(set(val))) for key, val in true_head.items()}
    return heads, relations, tails, true_tail, true_head


@dataclass
class ProteinGoInputFeatures:
    """
    A single set of feature of data for OntoProtein pretrain.
    """
    postive_protein_input_ids: List[int]
    postive_relation_ids: int
    postive_go_input_ids: Union[int, List[int]]
    negative_protein_input_ids: List[List[int]] = None
    negative_protein_attention_mask: Optional[List[int]] = None
    negative_relation_ids: List[int] = None
    negative_go_input_ids: List[Union[int, List[int]]] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"


@dataclass
class GoGoInputFeatures:
    """
    A single set of feature of data for Go-GO triplet in OntoProtein pretrain.
    """
    postive_go_head_input_ids: Union[int, List[int]]
    postive_relation_ids: int
    postive_go_tail_input_ids: Union[int, List[int]]
    negative_go_head_input_ids: List[Union[int, List[int]]] = None
    negative_relation_ids: List[int] = None
    negative_go_tail_input_ids: List[Union[int, List[int]]] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"


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
    

class ProteinGoDataset(Dataset):
    """
    Dataset for Protein-GO triplet.

    Args:
        data_dir: the diractory need contain pre-train datasets.
        use_seq: Whether or not to use the representation of protein sequence through encoder as entity embedding.
        use_desc: Whether or not to use the representation of Go term' description through encoder as entity embedding. 
                  Otherwise, using the embedding of Go term' entity in KE.
        protein_tokenizer: Tokenizer used to tokenize protein sequence.
        text_tokenizer: Tokenizer used to tokenize text.
        negative_sampling_fn: The strategy of negative sampling.
        num_neg_sample: the number of negative samples on one side. In other words, if set `sample_head` and `sample_tail`
                        to `True`, the total number of negative samples is 2*`num_neg_sample`.
        sample_head: Whether or not to construct negative sample pairs by fixing tail entity.
        sample_tail: Whether or not to construct negative sample pairs by fixing head entity.
        max_protein_seq_length: the max length of sequence. If set `None` to `max_seq_length`, It will dynamically set the max length
                        of sequence in batch to `max_seq_length`.
        max_text_seq_length: It need to set `max_text_seq_length` when using desciption of Go term to represent the Go entity.
    """
    def __init__(
        self,
        data_dir: str,
        use_seq: bool,
        use_desc: bool,
        protein_tokenizer: PreTrainedTokenizerBase = None,
        text_tokenizer: PreTrainedTokenizerBase = None,
        negative_sampling_fn = None,
        num_neg_sample: int = 1,
        sample_head: bool = False,
        sample_tail: bool = True,
        max_protein_seq_length: int = None,
        max_text_seq_length: int = None
    ):
        self.data_dir = data_dir
        self.use_seq = use_seq
        self.use_desc = use_desc
        self._load_data()

        self.protein_tokenizer = protein_tokenizer
        self.text_tokenizer = text_tokenizer
        self.negative_sampling_fn = negative_sampling_fn
        self.num_neg_sample = num_neg_sample
        self.sample_head = sample_head
        self.sample_tail = sample_tail
        self.max_protein_seq_length = max_protein_seq_length
        self.max_text_seq_length = max_text_seq_length
    
    def _load_data(self):
        self.go2id = [line.rstrip('\n') for line in open(os.path.join(self.data_dir, 'go2id.txt'), 'r')]
        self.relation2id = [line.rstrip('\n') for line in open(os.path.join(self.data_dir, 'relation2id.txt'), 'r')]
        self.num_go_terms = len(self.go2id)
        self.num_relations = len(self.relation2id)

        self.go_types = {idx: line.rstrip('\n') for idx, line in enumerate(open(os.path.join(self.data_dir, 'go_type.txt'), 'r'))}
        self.protein_seq = [line.rstrip('\n') for line in open(os.path.join(self.data_dir, 'protein_seq.txt'), 'r')]
        self.num_proteins = len(self.protein_seq)

        if self.use_desc:
            self.go_descs = {idx: line.rstrip('\n') for idx, line in enumerate(open(os.path.join(self.data_dir, 'go_def.txt'), 'r'))}
        
        # split go term according to ontology type.
        self.go_terms_type_dict = _split_go_by_type(self.go_types)

        # TODO: now adopt simple strategy of negative sampling. Wait to update.
        # for negative sample.
        self.protein_heads, self.pg_relations, self.go_tails, self.true_tail, self.true_head = get_triplet_data(
            data_path=os.path.join(self.data_dir, 'protein_go_train_triplet.txt')
        )
        
    def __getitem__(self, index):
        protein_head_id, relation_id, go_tail_id = self.protein_heads[index], self.pg_relations[index], self.go_tails[index]

        protein_input_ids = protein_head_id
        # use sequence.
        if self.use_seq:
            # tokenize protein sequence.
            protein_head_seq = list(self.protein_seq[protein_head_id])
            if self.max_protein_seq_length is not None:
                protein_head_seq = protein_head_seq[:self.max_protein_seq_length]
            protein_input_ids = self.protein_tokenizer.encode(list(protein_head_seq))

        go_tail_type = self.go_types[go_tail_id]
        go_input_ids = go_tail_id
        if self.use_desc:
            go_desc = self.go_descs[go_tail_id]
            go_input_ids = self.text_tokenizer.encode(go_desc, max_length=self.max_text_seq_length, truncation=True, padding='max_length')

        negative_protein_input_ids_list = []
        negative_relation_ids_list = []
        negative_go_input_ids_list = []

        if self.sample_tail:
            tail_negative_samples = self.negative_sampling_fn(
                cur_entity=(protein_head_id, relation_id),
                num_neg_sample=self.num_neg_sample,
                true_triplet=self.true_tail,
                num_entity=None,
                go_terms=self.go_terms_type_dict[go_tail_type]
            )

            for neg_go_id in tail_negative_samples:
                neg_go_input_ids = neg_go_id
                if self.use_desc:
                    neg_go_desc = self.go_descs[neg_go_id]
                    neg_go_input_ids = self.text_tokenizer.encode(neg_go_desc, max_length=self.max_text_seq_length, truncation=True, padding='max_length')

                negative_protein_input_ids_list.append(protein_input_ids)
                negative_relation_ids_list.append(relation_id)
                negative_go_input_ids_list.append(neg_go_input_ids)

        if self.sample_head:
            head_negative_samples = self.negative_sampling_fn(
                cur_entity=(relation_id, go_tail_id),
                num_neg_sample=self.num_neg_sample,
                true_triplet=self.true_head,
                num_entity=self.num_proteins,
                go_terms=None,
            )

            for neg_protein_id in head_negative_samples:
                neg_protein_input_ids = neg_protein_id
                if self.use_seq:
                    neg_protein_seq = list(self.protein_heads[neg_protein_id])
                    if self.max_protein_seq_length is not None:
                        neg_protein_seq = neg_protein_seq[:self.max_protein_seq_length]
                    neg_protein_input_ids = self.protein_tokenizer.encode(neg_protein_seq)

                negative_protein_input_ids_list.append(neg_protein_input_ids)
                negative_relation_ids_list.append(relation_id)
                negative_go_input_ids_list.append(go_input_ids)

        assert len(negative_protein_input_ids_list) == len(negative_relation_ids_list)
        assert len(negative_relation_ids_list) == len(negative_go_input_ids_list)

        return ProteinGoInputFeatures(
            postive_protein_input_ids=protein_input_ids,
            postive_relation_ids=relation_id,
            postive_go_input_ids=go_input_ids,
            negative_protein_input_ids=negative_protein_input_ids_list,
            negative_relation_ids=negative_relation_ids_list,
            negative_go_input_ids=negative_go_input_ids_list
        )

    def __len__(self):
        assert len(self.protein_heads) == len(self.pg_relations)
        assert len(self.pg_relations) == len(self.go_tails)

        return len(self.protein_heads)

    def get_num_go_terms(self):
        return len(self.go_types)

    def get_num_protein_go_relations(self):
        return len(list(set(self.pg_relations)))


class GoGoDataset(Dataset):
    """
    Dataset used for Go-Go triplet.

    Args:
        data_dir: the diractory need contain pre-train datasets.
        use_desc: Whether or not to use the representation of Go term' description through encoder as entity embedding. 
                  Otherwise, using the embedding of Go term' entity in KE.
        text_tokenizer: Tokenizer used for tokenize the description of Go term.
        negative_sampling_fn: the strategy of negative sampling.
        num_neg_sample: the number of negative samples on one side. In other words, if set `sample_head` and `sample_tail`
                        to `True`, the total number of negative samples is 2*`num_neg_sample`.
        sample_head: Whether or not to construct negative sample pairs by fixing tail entity.
        sample_tail: Whether or not to construct negative sample pairs by fixing head entity.
        max_text_seq_length: It need to set `max_text_seq_length` when using desciption of Go term to represent the Go entity.
    """

    def __init__(
        self,
        data_dir: str,
        use_desc: bool = False,
        text_tokenizer: PreTrainedTokenizerBase = None,
        negative_sampling_fn = None,
        num_neg_sample: int = 1,
        sample_head: bool = True,
        sample_tail: bool = True,
        max_text_seq_length: int = None
    ):
        self.data_dir = data_dir
        self.use_desc = use_desc
        self.text_tokenizer = text_tokenizer
        self.negative_sampling_fn = negative_sampling_fn
        self.num_neg_sample = num_neg_sample
        self.sample_head = sample_head
        self.sample_tail = sample_tail
        self.max_text_seq_length = max_text_seq_length
        self._load_data()

    def _load_data(self):
        self.go2id = [line.rstrip('\n') for line in open(os.path.join(self.data_dir, 'go2id.txt'), 'r')]
        self.relation2id = [line.rstrip('\n') for line in open(os.path.join(self.data_dir, 'relation2id.txt'), 'r')]
        self.num_go_terms = len(self.go2id)
        self.num_relations = len(self.relation2id)

        self.go_types = {idx: line.rstrip('\n') for idx, line in enumerate(open(os.path.join(self.data_dir, 'go_type.txt'), 'r'))}
        if self.use_desc:
            self.go_descs = {idx: line.rstrip('\n') for idx, line in enumerate(open(os.path.join(self.data_dir, 'go_def.txt'), 'r'))}

        # split go term according to ontology type.
        # same negative sampling strategy in `ProteinGODataset`
        self.go_terms_type_dict = _split_go_by_type(self.go_types)
        self.go_heads, self.gg_relations, self.go_tails, self.true_tail, self.true_head = get_triplet_data(
            data_path=os.path.join(self.data_dir, 'go_go_triplet.txt')
        )

    def __getitem__(self, index):
        go_head_id, relation_id, go_tail_id = self.go_heads[index], self.gg_relations[index], self.go_tails[index]

        go_head_type = self.go_types[go_head_id]
        go_tail_type = self.go_types[go_tail_id]
        go_head_input_ids = go_head_id
        go_tail_input_ids = go_tail_id
        if self.use_desc:
            go_head_desc = self.go_descs[go_head_id]
            go_tail_desc = self.go_descs[go_tail_id]
            go_head_input_ids = self.text_tokenizer.encode(go_head_desc, padding='max_length', truncation=True, max_length=self.max_text_seq_length)
            go_tail_input_ids = self.text_tokenizer.encode(go_tail_desc, padding='max_length', truncation=True, max_length=self.max_text_seq_length)
        
        negative_go_head_input_ids_list = []
        negative_relation_ids_list = []
        negative_go_tail_input_ids_list = []

        if self.sample_tail:
            tail_negative_samples = self.negative_sampling_fn(
                cur_entity=(go_head_id, relation_id),
                num_neg_sample=self.num_neg_sample,
                true_triplet=self.true_tail,
                num_entity=None,
                go_terms=self.go_terms_type_dict[go_tail_type]
            )

            for neg_go_id in tail_negative_samples:
                neg_go_input_ids = neg_go_id
                if self.use_desc:
                    neg_go_desc = self.go_descs[neg_go_id]
                    neg_go_input_ids = self.text_tokenizer.encode(neg_go_desc, max_length=self.max_text_seq_length, truncation=True, padding='max_length')

                negative_go_head_input_ids_list.append(go_head_input_ids)
                negative_relation_ids_list.append(relation_id)
                negative_go_tail_input_ids_list.append(neg_go_input_ids)

        if self.sample_head:
            head_negative_samples = self.negative_sampling_fn(
                cur_entity=(relation_id, go_tail_id),
                num_neg_sample=self.num_neg_sample,
                true_triplet=self.true_head,
                num_entity=None,
                go_terms=self.go_terms_type_dict[go_head_type]
            )

            for neg_go_id in head_negative_samples:
                neg_go_input_ids = neg_go_id
                if self.use_desc:
                    neg_go_desc = self.go_descs[neg_go_id]
                    neg_go_input_ids = self.text_tokenizer.encode(neg_go_desc, max_length=self.max_text_seq_length, truncation=True, padding='max_length')
                
                negative_go_head_input_ids_list.append(neg_go_input_ids)
                negative_relation_ids_list.append(relation_id)
                negative_go_tail_input_ids_list.append(go_tail_input_ids)

        assert len(negative_go_head_input_ids_list) == len(negative_relation_ids_list)
        assert len(negative_relation_ids_list) == len(negative_go_tail_input_ids_list)

        return GoGoInputFeatures(
            postive_go_head_input_ids=go_head_input_ids,
            postive_relation_ids=relation_id,
            postive_go_tail_input_ids=go_tail_input_ids,
            negative_go_head_input_ids=negative_go_head_input_ids_list,
            negative_relation_ids=negative_relation_ids_list,
            negative_go_tail_input_ids=negative_go_tail_input_ids_list
        )

    def __len__(self):
        assert len(self.go_heads) == len(self.gg_relations)
        assert len(self.gg_relations) == len(self.go_tails)

        return len(self.go_heads)

    def get_num_go_terms(self):
        return len(self.go_types)

    def get_num_go_go_relations(self):
        return len(list(set(self.gg_relations)))


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
        data_dir: str,
        seq_data_path: str = None,
        tokenizer: PreTrainedTokenizerBase = None,
        in_memory: bool=True,
        max_protein_seq_length: int = None
    ):
        self.data_dir = data_dir
        self.seq_data_path = seq_data_path

        # self.env = lmdb.open(os.path.join(data_dir, seq_data_path), readonly=True)
        
        # with self.env.begin(write=False) as txn:
        #     self.num_examples = pkl.loads(txn.get(b'num_examples'))

        # self.in_memory = in_memory
        # if in_memory:
        #     cache = [None] * self.num_examples
        #     self.cache = cache
        self.protein_seq = [line.rstrip('\n') for line in open(os.path.join(self.data_dir, 'protein_seq.txt'), 'r')]

        self.tokenizer = tokenizer
        self.max_protein_seq_length = max_protein_seq_length
        
    def __getitem__(self, index):
        # if self.in_memory and self.cache[index] is not None:
        #     item = self.cache[index]
        # else:
        #     with self.env.begin(write=False) as txn:
        #         item = pkl.loads(txn.get(str(index).encode()))
        #     if self.in_memory:
        #         self.cache[index] = item
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

