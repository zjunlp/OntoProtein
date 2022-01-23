import os
import lmdb
import pickle as pkl
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer


def pad_sequences(sequences, constant_value=0, dtype=None):
    batch_size = len(sequences)
    shape = [batch_size] + np.max([seq.shape for seq in sequences], axis=0).tolist()

    if dtype == None:
        dtype = sequences[0].dtype

    array = np.full(shape, constant_value, dtype)

    for arr, seq in zip(array, sequences):
        arrslice = tuple(slice(dim) for dim in seq.shape)
        arr[arrslice] = seq

    return array


class LMDBDataset(Dataset):
    def __init__(self, data_file, in_memory):
        env = lmdb.open(data_file, max_readers=1, readonly=True,
                        lock=False, readahead=False, meminit=False)
        
        with env.begin(write=False) as txn:
            num_examples = pkl.loads(txn.get(b'num_examples'))

        if in_memory:
            cache = [None] * num_examples
            self._cache = cache
        
        self._env = env
        self._in_memory = in_memory
        self._num_examples = num_examples

    def __len__(self):
        return self._num_examples

    def __getitem__(self, index):
        if self._in_memory and self._cache[index] is not None:
            item = self._cache[index]
        else:
            with self._env.begin(write=False) as txn:
                item = pkl.loads(txn.get(str(index).encode()))
                if 'id' not in item:
                    item['id'] = str(index)
                if self._in_memory:
                    self._cache[index] = item
        return item


class DataProcessor:
    """Base class for data converters for biological tasks data sets."""
    def get_train_examples(self, data_dir):
        """Gets a collection of :class:`InputExample` for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of :class:`InputExample` for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of :class:`InputExample` for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    
class SecondaryStructureProcessor(DataProcessor):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
    
    def get_train_examples(self, data_dir, target='ss3', in_memory=True):
        dataset = SecondaryStructureDataset(data_dir, split='train', tokenizer=self.tokenizer, in_memory=in_memory, target=target)
        return dataset
    
    def get_dev_examples(self, data_dir, target='ss3', in_memory=True):
        dataset = SecondaryStructureDataset(data_dir, split='valid', tokenizer=self.tokenizer, target=target, in_memory=in_memory)
        return dataset

    def get_test_examples(self, data_dir, data_cat, target='ss3', in_memory=True):
        dataset = SecondaryStructureDataset(data_dir, split=data_cat, tokenizer=self.tokenizer, target=target, in_memory=in_memory)
        return dataset

    def get_labels(self, target='ss3'):
        if target == 'ss3':
            return list(range(3))
        elif target == 'ss8':
            return list(range(8))
        else:
            raise Exception("target not supported.")
    

class SecondaryStructureDataset(Dataset):
    def __init__(
        self,
        data_path,
        split,
        tokenizer,
        in_memory,
        target='ss3',
        max_length=512
    ):
        self.tokenizer = tokenizer
        data_file = f'secondary_structure/secondary_structure_{split}.lmdb'
        self.data = LMDBDataset(data_file=os.path.join(data_path, data_file), in_memory=in_memory)
        self.target = target

        self.ignore_index: int = -100
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        protein_seq = list(item['primary'])
        labels = item[self.target]
        disorder = item['disorder']
        if len(protein_seq) > self.max_length:
            protein_seq = protein_seq[:self.max_length] 
            labels = labels[:self.max_length]
            disorder = disorder[:self.max_length]

        input_ids = np.array(self.tokenizer.encode(protein_seq), dtype=np.int64)
        attention_mask = np.ones_like(input_ids)

        # mask label of index which disorder is 0.
        # 'transformers' implement 'LabelSmoother', which ignore index of '-100'.
        labels = np.array([l if mask == 1 else self.ignore_index for l, mask in zip(labels, disorder)], dtype=np.int64)
        # pad labels due to additional special tokens.
        labels = np.pad(labels, (1, 1), 'constant', constant_values=self.ignore_index)
        
        return input_ids, attention_mask, labels

    def collate_fn(self, batch):
        input_ids, attention_mask, labels = tuple(zip(*batch))
        # dynamic max sequence length.
        input_ids = torch.from_numpy(pad_sequences(input_ids, constant_value=self.tokenizer.pad_token_id))
        attention_mask = torch.from_numpy(pad_sequences(attention_mask, constant_value=0))
        labels = torch.from_numpy(pad_sequences(labels, constant_value=self.ignore_index))

        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

        return inputs


output_modes_mapping = {
    'ssp': 'token-level-classification'
}


bt_processors = {
    'ssp': SecondaryStructureProcessor,

}