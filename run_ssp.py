import os
import logging
from dataclasses import field, dataclass
from typing import Optional
import torch
from transformers import set_seed
from transformers import HfArgumentParser, TrainingArguments
from transformers import AutoConfig, BertTokenizer
# from transformers import Trainer

from src.benchmark.dataset import bt_processors, output_modes_mapping
from src.benchmark.models import model_fn_mapping
from src.benchmark.trainer import Trainer
from src.benchmark.metrics import build_compute_metrics_fn

import warnings
warnings.filterwarnings("ignore")


logger = logging.getLogger(__name__)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
DEVICE="cuda"


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
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


@dataclass
class DynamicTrainingArguments(TrainingArguments):
    # For ensemble
    array_id: int = field(
        default=-1,
        metadata={"help": "Array ID (contains seed and hyper-paramter search) to idenfity the model"}
    )

    model_id: int = field(
        default=-1,
        metadata={"help": "Model ID (contains template information) to identify the model"}
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

    # Turn off train/test
    no_train: bool = field(
        default=False,
        metadata={"help": "No training"}
    )
    no_predict: bool = field(
        default=False,
        metadata={"help": "No test"}
    )
    evaluate_during_training: bool = field(
        default=True,
        metadata={"help": "evaluate during training."}
    )


@dataclass
class BTDataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: str = field(metadata={"help": "The name of the task to train on: " + ", ".join(bt_processors.keys())})
    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."}
    )
    data_cat: str = field(
        metadata={"help": "A few tasks exist multiple test dataset."}
    )
    target: str = field(
        default='ss3',
        metadata={"help": "Secondary Strcuture Prediction task have multiple target."}
    )
    max_seq_length: int = field(
        default=-1,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
            "default is -1 due to the adoption of dynamic batching"
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    def __post_init__(self):
        self.task_name = self.task_name.lower()


def main():
    parser = HfArgumentParser((ModelArguments, BTDataTrainingArguments, DynamicTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.no_train:
        training_args.do_train = False
    if training_args.no_predict:
        training_args.do_predict = False
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    # Check save path
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(f"Output directory ({training_args.output_dir}) already exists.")

    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        DEVICE,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )

    logger.info("Training/evaluation parameters %s", training_args)

    set_seed(training_args.seed)

    try:
        output_mode = output_modes_mapping[data_args.task_name]
        logger.info("Task name: {}, output mode: {}".format(data_args.task_name, output_mode))
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    # Load dataset
    tokenizer = BertTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    processor = bt_processors[data_args.task_name](tokenizer=tokenizer)
    # For classification task, num labels is determined by specific tasks
    # For regression task, num labels is 1.
    num_labels = len(processor.get_labels())
    train_dataset = (
        processor.get_train_examples(data_dir=data_args.data_dir, target=data_args.target)
    )
    eval_dataset = (
        processor.get_dev_examples(data_dir=data_args.data_dir, target=data_args.target)
    )
    test_dataset = (
        processor.get_test_examples(data_dir=data_args.data_dir, target=data_args.target, data_cat=data_args.data_cat)
        if training_args.do_predict
        else None
    )

    model_fn = model_fn_mapping[data_args.task_name]
    model = model_fn.from_pretrained(
        model_args.model_name_or_path,
        num_labels=num_labels,
    )

    # # freeze pretrained model
    # for param in model.bert.parameters():
    #     param.requires_grad = False

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=build_compute_metrics_fn(data_args.task_name, output_type=output_mode),
        data_collator=train_dataset.collate_fn
    )

    # Training
    if training_args.do_train:
        trainer.train()

        tokenizer.save_pretrained(training_args.output_dir)
        torch.save(model_args, os.path.join(training_args.output_dir, "model_args.bin"))
        torch.save(data_args, os.path.join(training_args.output_dir, "data_args.bin"))

    # Prediction
    if training_args.do_predict:
        logger.info("**** Test ****")

        trainer.compute_metrics = build_compute_metrics_fn(data_args.task_name, output_type=output_mode)
        output = trainer.evaluate(eval_dataset=test_dataset)
        test_result = output.metrics
        output_test_file = os.path.join(
            training_args.output_dir, f"{data_args.data_cat}_results_{data_args.task_name}.txt"
        )
        with open(output_test_file, "w") as writer:
            logger.info("***** Test results {} *****".format(data_args.task_name))
            for key, value in test_result.items():
                logger.info("  %s = %s", key, value)
                writer.write("%s = %s\n" % (key, value))

if __name__ == '__main__':
    main()