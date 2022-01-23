# import time
# import torch
# from torch.utils.data import DataLoader
# from transformers import BertTokenizer, BertForSequenceClassification
# from transformers import AutoConfig, BertForMaskedLM
# from dataset import ProteinGoDataset, ProteinSeqDataset, GoGoDataset
# from sampling import negative_sampling_strategy
# from dataloader import DataCollatorForGoGo, DataCollatorForProteinGo, DataCollatorForLanguageModeling


if __name__ == '__main__':
    # pg_dataset = ProteinGODataset('data/pretrain_data')

    # protein_tokenizer = BertTokenizer.from_pretrained('data/model_data/ProtBERT')
    # protein_seq_datasets = ProteinSeqDataset(
    #     data_dir='data/pretrain_data',
    #     seq_data_path='swiss_seq',
    #     tokenizer=protein_tokenizer
    # )

    # protein_seq_data_collator = DataCollatorForLanguageModeling(tokenizer=protein_tokenizer, are_protein_length_same=False)
    # protein_seq_dataloader = DataLoader(
    #     protein_seq_datasets, 
    #     batch_size=512, 
    #     collate_fn=protein_seq_data_collator,
    # )

    # start_time = time.time()
    # for item in protein_seq_dataloader:
    #     print(item)
    #     break
    # print(time.time() - start_time)

    # protein_go_dataset = ProteinGoDataset(
    #     data_dir='data/pretrain_data',
    #     use_desc=False,
    #     protein_tokenizer=protein_tokenizer,
    #     negative_sampling_fn=negative_sampling_strategy['simple_random'],
    #     num_neg_sample=2,
    #     sample_head=False,
    #     sample_tail=True,
    # )

    # protein_go_data_collator = DataCollatorForProteinGo(protein_tokenizer=protein_tokenizer, are_protein_length_same=False)
    # protein_go_dataloader = DataLoader(
    #     dataset=protein_go_dataset,
    #     batch_size=512,
    #     collate_fn=protein_go_data_collator,
    #     num_workers=4
    # )

    # start_time = time.time()
    # for item in protein_go_dataloader:
    #     print(item)
    #     break
    # print(time.time() - start_time)
    # # print(len(protein_go_dataset) // 512)

    # go_go_dataset = GoGoDataset(
    #     data_dir='data/pretrain_data',
    #     use_desc=False,
    #     negative_sampling_fn=negative_sampling_strategy['simple_random'],
    #     num_neg_sample=1,
    #     sample_head=True,
    #     sample_tail=True,
    # )

    # go_go_data_collator = DataCollatorForGoGo()
    # go_go_dataloader = DataLoader(
    #     dataset=go_go_dataset,
    #     batch_size=512,
    #     collate_fn=go_go_data_collator,
    #     num_workers=4
    # )

    # start_time = time.time()
    # for item in go_go_dataloader:
    #     print(item)
    #     break
    # print(time.time() - start_time)

    # model = BertForSequenceClassification.from_pretrained('data/model_data/ProtBERT')

    # config = OntoProteinConfig.from_pretrained()

    import torch

    # class TestModel(nn.Module):
    #     def __init__(self, config):
    #         super().__init__()

    #         self.encoder = BertForMaskedLM(config)
        
    #     def forward(self, x):
    #         return x

    #     @classmethod
    #     def from_pretrained(cls, model_path):
    #         config = AutoConfig.from_pretrained(model_path)
    #         model = cls(config)

    #         model.encoder = BertForMaskedLM.from_pretrained(model_path)
            
    #         return model

    # model = TestModel.from_pretrained('data/model_data/ProtBERT')
    # print(model.state_dict().keys())

    from transformers import BertForSequenceClassification

    model = BertForSequenceClassification.from_pretrained('data/output_data/checkpoint-1171/protein')
    