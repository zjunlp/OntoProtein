import os
import logging
from transformers import HfArgumentParser, set_seed, logging
from transformers import BertTokenizer, AutoTokenizer

from src.models import OntoProteinPreTrainedModel
from src.trainer import OntoProteinTrainer
from src.sampling import negative_sampling_strategy
from src.dataset import ProteinSeqDataset, ProteinGoDataset, GoGoDataset
from src.dataloader import DataCollatorForGoGo, DataCollatorForLanguageModeling, DataCollatorForProteinGo
from src.training_args import OntoProteinModelArguments, OntoProteinDataTrainingArguments, OntoProteinTrainingArguments

logger = logging.get_logger(__name__)
DEVICE = 'cuda'


def main():
    parser = HfArgumentParser((OntoProteinTrainingArguments, OntoProteinDataTrainingArguments, OntoProteinModelArguments))
    training_args, data_args, model_args = parser.parse_args_into_dataclasses()

    # check output_dir
    os.makedirs(training_args.output_dir, exist_ok=True)

    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        DEVICE,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )

    logger.info(f"Training parameters: {training_args}")

    set_seed(training_args.seed)

    # default BertTokenizer
    if model_args.protein_model_file_name:
        protein_tokenizer = BertTokenizer.from_pretrained(model_args.protein_model_file_name)
    else:
        raise ValueError("Need provide protein tokenizer config path.")

    text_tokenizer = None
    if model_args.text_model_file_name and training_args.use_desc:
        text_tokenizer = AutoTokenizer.from_pretrained('data/model_data/PubMedBERT')

    # Load dataset
    protein_seq_dataset = None
    protein_go_dataset = None
    go_go_dataset = None

    # negative sampling strategy
    # TODO: seperate into Protein-side and Go-side negative sampling strategy.
    negative_sampling_fn = negative_sampling_strategy[data_args.negative_sampling_fn]

    if data_args.model_protein_seq_data:
        protein_seq_dataset = ProteinSeqDataset(
            data_dir=data_args.pretrain_data_dir,
            seq_data_path=data_args.protein_seq_data_file_name,
            tokenizer=protein_tokenizer,
            max_protein_seq_length=data_args.max_protein_seq_length,
        )

    if data_args.model_protein_go_data:
        protein_go_dataset = ProteinGoDataset(
            data_dir=data_args.pretrain_data_dir,
            use_desc=training_args.use_desc,
            use_seq=training_args.use_seq,
            protein_tokenizer=protein_tokenizer,
            text_tokenizer=text_tokenizer,
            negative_sampling_fn=negative_sampling_fn,
            num_neg_sample=training_args.num_protein_go_neg_sample,
            sample_head=data_args.protein_go_sample_head,
            sample_tail=data_args.protein_go_sample_tail,
            max_protein_seq_length=data_args.max_protein_seq_length,
            max_text_seq_length=data_args.max_text_seq_length
        )

    if data_args.model_go_go_data:
        go_go_dataset = GoGoDataset(
            data_dir=data_args.pretrain_data_dir,
            text_tokenizer=text_tokenizer,
            use_desc=training_args.use_desc,
            negative_sampling_fn=negative_sampling_fn,
            num_neg_sample=training_args.num_go_go_neg_sample,
            sample_head=data_args.go_go_sample_head,
            sample_tail=data_args.go_go_sample_tail,
            max_text_seq_length=data_args.max_text_seq_length,
        )

    # Ontology statistics
    num_relations = protein_go_dataset.num_relations
    num_go_terms = protein_go_dataset.num_go_terms
    num_proteins = protein_go_dataset.num_proteins

    # init data collator
    are_protein_length_same = False
    protein_seq_data_collator = DataCollatorForLanguageModeling(tokenizer=protein_tokenizer, are_protein_length_same=are_protein_length_same)
    protein_go_data_collator = DataCollatorForProteinGo(protein_tokenizer=protein_tokenizer, text_tokenizer=text_tokenizer, are_protein_length_same=are_protein_length_same)
    go_go_data_collator = DataCollatorForGoGo(tokenizer=text_tokenizer)

    model = OntoProteinPreTrainedModel.from_pretrained(
        protein_model_path=model_args.protein_model_file_name,
        onto_model_path=model_args.text_model_file_name,
        model_args=model_args,
        training_args=training_args,
        num_relations=num_relations,
        num_go_terms=num_go_terms,
        num_proteins=num_proteins,
    )

    # freeze pretrained model
    # if model_args.protein_encoder_cls == 'bert':
    #     for param in model.protein_lm.bert.parameters():
    #         param.requires_grad = False

    # prepare Trainer
    trainer = OntoProteinTrainer(
        model=model,
        args=training_args,
        protein_seq_dataset=protein_seq_dataset,
        protein_go_dataset=protein_go_dataset,
        go_go_dataset=go_go_dataset,
        protein_seq_data_collator=protein_seq_data_collator,
        protein_go_data_collator=protein_go_data_collator,
        go_go_data_collator=go_go_data_collator
    )

    # Pretraining
    if training_args.do_train:
        trainer.train()


if __name__ == "__main__":
    main()