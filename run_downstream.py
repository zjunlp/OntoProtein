import os
from typing import Optional

from dataclasses import dataclass, field
from transformers import HfArgumentParser, TrainingArguments, BertTokenizerFast, set_seed, Trainer
import logging

from src.models import model_mapping, load_adam_optimizer_and_scheduler
from src.datasets import dataset_mapping, output_modes_mapping
from src.metrics import build_compute_metrics_fn
from src.trainer import OntoProteinTrainer

import warnings
warnings.filterwarnings("ignore")

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
logger = logging.getLogger(__name__)
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# DEVICE = "cuda"


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )

    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )

    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

    mean_output: bool = field(
        default=True, metadata={"help": "output of bert, use mean output or pool output"}
    )

    optimizer: str = field(
        default="AdamW",
        metadata={"help": "use optimizer: AdamW(True) or Adam(False)."}
    )

    frozen_bert: bool = field(
        default=False,
        metadata={"help": "frozen bert model."}
    )


@dataclass
class DynamicTrainingArguments(TrainingArguments):
    # For ensemble
    save_strategy: str = field(
        default='steps',
        metadata={"help": "The checkpoint save strategy to adopt during training."}
    )

    save_steps: int = field(
        default=500,
        metadata={"help": " Number of updates steps before two checkpoint saves"}
    )

    evaluation_strategy: str = field(
        default='steps',
        metadata={"help": "The evaluation strategy to adopt during training."}
    )

    eval_steps: int = field(
        default=100,
        metadata={"help": "Number of update steps between two evaluations"}
    )

    save_logit: bool = field(
        default=False,
        metadata={"help": "Save test file logit with name $TASK-$MODEL_ID-$ARRAY_ID.npy"}
    )

    save_logit_dir: str = field(
        default=None,
        metadata={"help": "Where to save the prediction result"}
    )

    # Regularization
    fix_layers: int = field(
        default=0,
        metadata={"help": "Fix bottom-n layers when optimizing"}
    )

    evaluate_during_training: bool = field(
        default=True,
        metadata={"help": "evaluate during training."}
    )
    save_total_limit: int = field(
        default=3,
        metadata={"help": "If a value is passed, will limit the total amount of checkpoints."}
    )
    # resume_from_checkpoint = True
    fp16 = True


@dataclass
class BTDataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: str = field(metadata={"help": "The name of the task to train on: " + ", ".join(dataset_mapping.keys())})
    data_dir: str = field(
        default=None,
        metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    def __post_init__(self):
        self.task_name = self.task_name.lower()


def main():
    parser = HfArgumentParser((ModelArguments, BTDataTrainingArguments, DynamicTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    # Check save path
    # if (
    #         os.path.exists(training_args.output_dir)
    #         and os.listdir(training_args.output_dir)
    #         and training_args.do_train
    #         and not training_args.overwrite_output_dir
    # ):
    #     raise ValueError(f"Output directory ({training_args.output_dir}) already exists.")

    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s",
        training_args.local_rank,
        # DEVICE,
        training_args.n_gpu,
        bool(training_args.local_rank != -1)
    )

    logger.info("Training/evaluation parameters %s", training_args)

    set_seed(training_args.seed)

    try:
        output_mode = output_modes_mapping[data_args.task_name]
        logger.info("Task name: {}, output mode: {}".format(data_args.task_name, output_mode))
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    # Load dataset
    tokenizer = BertTokenizerFast.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        do_lower_case=False
    )
    processor = dataset_mapping[data_args.task_name](tokenizer=tokenizer)
    # For classification task, num labels is determined by specific tasks
    # For regression task, num labels is 1.
    num_labels = len(processor.get_labels())
    train_dataset = (
        processor.get_train_examples(data_dir=data_args.data_dir)
    )
    eval_dataset = (
        processor.get_dev_examples(data_dir=data_args.data_dir)
    )
    if data_args.task_name == 'remote_homology':
        test_fold_dataset = (
            processor.get_test_examples(data_dir=data_args.data_dir, data_cat='test_fold_holdout')
        )
        test_family_dataset = (
            processor.get_test_examples(data_dir=data_args.data_dir, data_cat='test_family_holdout')
        )
        test_superfamily_dataset = (
            processor.get_test_examples(data_dir=data_args.data_dir, data_cat='test_superfamily_holdout')
        )
    elif data_args.task_name == 'ss3' or data_args.task_name == 'ss8':
        print(data_args.task_name + ' test_dataset')
        cb513_dataset = (
            processor.get_test_examples(data_dir=data_args.data_dir, data_cat='cb513')
        )
        ts115_dataset = (
            processor.get_test_examples(data_dir=data_args.data_dir, data_cat='ts115')
        )
        casp12_dataset = (
            processor.get_test_examples(data_dir=data_args.data_dir, data_cat='casp12')
        )
    else:
        test_dataset = (
            processor.get_test_examples(data_dir=data_args.data_dir, data_cat='test')
        )

    model_fn = model_mapping[data_args.task_name]

    model = model_fn.from_pretrained(
        model_args.model_name_or_path,
        num_labels=num_labels,
        mean_output=model_args.mean_output,
        gradient_checkpointing=False
    )

    if model_args.frozen_bert:
        unfreeze_layers = ['layer.29', 'bert.pooler', 'classifier']
        for name, parameters in model.named_parameters():
            parameters.requires_grad = False
            for tags in unfreeze_layers:
                if tags in name:
                    parameters.requires_grad = True
                    break

    if data_args.task_name == 'stability' or data_args.task_name == 'fluorescence':
        training_args.metric_for_best_model = "eval_spearmanr"
    elif data_args.task_name == 'remote_homology':
        training_args.metric_for_best_model = "eval_accuracy"
    else:
        pass

    if data_args.task_name == 'contact':
        # training_args.do_predict=False
        trainer = OntoProteinTrainer(
            # model_init=init_model,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=build_compute_metrics_fn(data_args.task_name, output_type=output_mode),
            data_collator=train_dataset.collate_fn,
            optimizers=load_adam_optimizer_and_scheduler(model, training_args, train_dataset) if model_args.optimizer=='Adam' else (None, None)
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=build_compute_metrics_fn(data_args.task_name, output_type=output_mode),
            data_collator=train_dataset.collate_fn,
            optimizers=load_adam_optimizer_and_scheduler(model, training_args, train_dataset) if model_args.optimizer=='Adam' else (None, None)
        )

    # Training
    if training_args.do_train:
        # pass
        trainer.train()
        trainer.save_model(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)

    # Prediction
    logger.info("**** Test ****")

    # trainer.compute_metrics = metrics_mapping(data_args.task_name)
    if data_args.task_name == 'remote_homology':
        predictions_fold_family, input_ids_fold_family, metrics_fold_family = trainer.predict(test_fold_dataset)
        predictions_family_family, input_ids_family_family, metrics_family_family = trainer.predict(test_family_dataset)
        predictions_superfamily_family, input_ids_superfamily_family, metrics_superfamily_family = trainer.predict(test_superfamily_dataset)
        print("metrics_fold: ", metrics_fold_family)
        print("metrics_family: ", metrics_family_family)
        print("metrics_superfamily: ", metrics_superfamily_family)
    elif data_args.task_name == 'ss8' or data_args.task_name == 'ss3':
        predictions_cb513, input_ids_cb513, metrics_cb513 = trainer.predict(cb513_dataset)
        predictions_ts115, input_ids_ts115, metrics_ts115 = trainer.predict(ts115_dataset)
        predictions_casp12, input_ids_casp12, metrics_casp12 = trainer.predict(casp12_dataset)
        print("cb513: ", metrics_cb513)
        print("ts115: ", metrics_ts115)
        print("casp12: ", metrics_casp12)
    else:
        predictions_family, input_ids_family, metrics_family = trainer.predict(test_dataset)
        print("metrics", metrics_family)


if __name__ == '__main__':
    main()
