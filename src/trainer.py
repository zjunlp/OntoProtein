import json
import os
import math
import collections
import time
from tqdm import trange
from packaging import version
from typing import Optional, Tuple, Union, Dict, Any, List

import torch
import torch.nn as nn
from torch.utils.data import RandomSampler
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import Trainer, PreTrainedModel, logging
from transformers.deepspeed import deepspeed_init
from transformers.training_args import ShardedDDPOption, ParallelMode
from transformers.trainer_pt_utils import get_parameter_names, IterableDatasetShard
from transformers.optimization import Adafactor, AdamW
from transformers.trainer_callback import TrainerState
from transformers.file_utils import is_apex_available, is_sagemaker_mp_enabled

from src.dataset import GoGoDataset, ProteinGoDataset, ProteinSeqDataset
from src.dataloader import DataCollatorForLanguageModeling, DataCollatorForGoGo, DataCollatorForProteinGo
from src.models import OntoModel, OntoProteinPreTrainedModel, OntoProteinKELoss, OntoProteinMLMLoss, BertForMaskedLM
from src.optimization import get_scheduler


logger = logging.get_logger(__name__)


# if is_apex_available():
#     from apex import amp


if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_torch_generator_available = True
    _is_native_amp_available = True
    from torch.cuda.amp import autocast

# TODO: distributed training (multi-GPU)
# Data parallelism: sharded_ddp
# Model parallelism: deepspeed


class OntoProteinTrainer(Trainer):
    """
    OntoProteinTrainer implement the pretraining of protein language model with knowledge injection, which class 
    inherits from `transformer.Trainer`.

    Args:
        model: The model to train, evaluate or use for prediction
        args: The hyper-parameters for training, which will default to a basic instance :class:`~transformers.TrainingArguments`
        protein_seq_dataset: The instance of :class:`ProteinSeqDataset` 
        protein_go_dataset: The instance of :class:`ProteinGoDataset`
        go_go_dataset: The instance of :class:`GoGoDataset`
        protein_seq_data_collator: Data collator for :obj:`protein_seq_dataset`
        protein_go_data_collator: Data collator for :obj:`protein_go_dataset`
        go_go_data_collator: Data collator for :obj:`go_go_dataset`
    """

    def __init__(
        self,
        model: Union[nn.Module, PreTrainedModel],
        args,
        protein_seq_dataset: ProteinSeqDataset = None,
        protein_go_dataset: ProteinGoDataset = None,
        go_go_dataset: GoGoDataset = None,
        protein_seq_data_collator: DataCollatorForLanguageModeling = None,
        protein_go_data_collator: DataCollatorForProteinGo = None,
        go_go_data_collator: DataCollatorForGoGo = None,
    ):
        super().__init__(
            model=model,
            args=args,
        )

        self.protein_seq_dataset = protein_seq_dataset
        self.protein_go_dataset = protein_go_dataset
        self.go_go_dataset = go_go_dataset
        self.protein_seq_data_collator = protein_seq_data_collator
        self.protein_go_data_collator = protein_go_data_collator
        self.go_go_data_collator = go_go_data_collator

        self.ke_loss_fn = OntoProteinKELoss(
            ke_lambda=self.args.ke_lambda, 
            max_score=self.args.ke_max_score,
            num_protein_go_neg_sample=self.args.num_protein_go_neg_sample,
            num_go_go_neg_sample=self.args.num_go_go_neg_sample,
            score_fn=self.args.ke_score_fn
        )

        self.mlm_loss_fn = OntoProteinMLMLoss(
            mlm_lambda=self.args.mlm_lambda
        )

    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Union["optuna.Trial", Dict[str, Any]] = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Rewrite '~transformers.Trainer.train'
        """

        args = self.args

        self.is_in_train = True

        # Keeping track whether we can len() on the train dataset.
        train_dataset_is_sized = isinstance(self.protein_seq_dataset, collections.abc.Sized) or isinstance(self.protein_go_dataset, collections.abc.Sized)

        # Dataloader
        protein_seq_dataloader, protein_go_dataloader, go_go_dataloader = self.get_train_dataloader()
        
        total_train_protein_seq_batch_size = args.train_protein_seq_batch_size * args.gradient_accumulation_steps * args.world_size
        total_train_protein_go_batch_size = args.train_protein_go_batch_size * args.gradient_accumulation_steps * args.world_size
        total_train_go_go_batch_size = args.train_go_go_batch_size * args.gradient_accumulation_steps * args.world_size

        if train_dataset_is_sized:
            num_protein_seq_update_steps_per_epoch = max(len(protein_seq_dataloader) // args.gradient_accumulation_steps, 1) if protein_seq_dataloader else -1
            num_protein_go_update_steps_per_epoch = max(len(protein_go_dataloader) // args.gradient_accumulation_steps, 1) if protein_go_dataloader else -1
            num_go_go_update_steps_per_epoch = max(len(go_go_dataloader) // args.gradient_accumulation_steps, 1) if go_go_dataloader else -1

            if args.max_steps > 0:
                max_protein_seq_steps = args.max_steps
                num_protein_seq_epochs = args.max_steps // num_protein_seq_update_steps_per_epoch + int(
                    args.max_steps % num_protein_seq_update_steps_per_epoch > 0
                ) if num_protein_seq_update_steps_per_epoch else 0
                num_protein_seq_train_samples = args.max_steps * total_train_protein_seq_batch_size

                max_protein_go_steps = args.max_steps
                num_protein_go_epochs = args.max_steps // num_protein_go_update_steps_per_epoch + int(
                    args.max_steps % num_protein_go_update_steps_per_epoch > 0
                ) if num_protein_go_update_steps_per_epoch else 0
                num_protein_go_train_samples = args.max_steps * total_train_protein_go_batch_size

                max_go_go_steps = args.max_steps
                num_go_go_epochs = args.max_steps // num_go_go_update_steps_per_epoch + int(
                    args.max_steps % num_go_go_update_steps_per_epoch > 0
                ) if num_go_go_update_steps_per_epoch else 0
                num_go_go_train_samples = args.max_steps * total_train_go_go_batch_size
            else:
                max_protein_seq_steps = math.ceil(args.num_protein_seq_epochs * num_protein_seq_update_steps_per_epoch)
                num_protein_seq_epochs = math.ceil(args.num_protein_seq_epochs)
                num_protein_seq_train_samples = len(self.protein_seq_dataset) * args.num_protein_seq_epochs
            
                max_protein_go_steps = math.ceil(args.num_protein_go_epochs * num_protein_go_update_steps_per_epoch)
                num_protein_go_epochs = math.ceil(args.num_protein_go_epochs)
                num_protein_go_train_samples = len(self.protein_go_dataset) * args.num_protein_go_epochs

                max_go_go_steps = math.ceil(args.num_go_go_epochs * num_go_go_update_steps_per_epoch)
                num_go_go_epochs = math.ceil(args.num_go_go_epochs)
                num_go_go_train_samples = len(self.go_go_dataset) * args.num_go_go_epochs
        else:
            raise NotImplementedError("Not support dataset which don't implement `__len__`.")
            
        delay_optimizer_creation = self.sharded_ddp is not None and self.sharded_ddp != ShardedDDPOption.SIMPLE

        # TODO: Only support same max steps of training on the three dataset at present.
        assert max_protein_seq_steps == max_protein_go_steps & max_protein_go_steps == max_go_go_steps, "Only support same max_steps on the three dataset"
        max_steps = max_protein_seq_steps
        if args.deepspeed:
            
            deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
                self, num_training_steps=max_steps, resume_from_checkpoint=resume_from_checkpoint
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
        elif not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)


        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        model = self._wrap_model(self.model_wrapped)

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # Train
        num_protein_seq_examples = (
            self.num_examples(protein_seq_dataloader) if train_dataset_is_sized else total_train_protein_seq_batch_size * max_steps
        )
        num_protein_go_examples = (
            self.num_examples(protein_go_dataloader) if train_dataset_is_sized else total_train_protein_go_batch_size * max_steps
        )
        num_go_go_examples = (
            self.num_examples(go_go_dataloader) if train_dataset_is_sized else total_train_go_go_batch_size * max_steps
        )

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_protein_seq_examples} | {num_protein_go_examples} | {num_go_go_examples}")
        logger.info(f"  Num Epochs = {num_protein_seq_epochs} | {num_protein_go_epochs} | {num_go_go_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_protein_seq_batch_size} | {total_train_protein_go_batch_size} | {total_train_go_go_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        tr_loss = torch.tensor(0.0).to(args.device)
        self.loss_recorder = []
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        if isinstance(protein_seq_dataloader, DataLoader) and isinstance(protein_seq_dataloader.sampler, DistributedSampler):
            protein_seq_dataloader.sampler.set_epoch(0)
            protein_go_dataloader.sampler.set_epoch(0)
            go_go_dataloader.sampler.set_epoch(0)

        protein_seq_iter = iter(protein_seq_dataloader) if protein_seq_dataloader else None
        protein_go_iter = iter(protein_go_dataloader) if protein_go_dataloader else None
        go_go_iter = iter(go_go_dataloader) if go_go_dataloader else None

        num_protein_seq_steps_per_epoch = max(len(protein_seq_dataloader), 1) if protein_seq_dataloader else -1
        num_protein_go_steps_per_epoch = max(len(protein_go_dataloader), 1) if protein_go_dataloader else -1
        num_go_go_steps_per_epoch = max(len(go_go_dataloader), 1) if go_go_dataloader else -1

        # record epoch for update of seed on dataloaders.
        cur_protein_seq_epoch = 0
        cur_protein_go_epoch = 0
        cur_go_go_epoch = 0

        train_iterator = range(
            epochs_trained, max_steps
        )

        for step in train_iterator:
            # update the seed of dataloader
            if num_protein_seq_steps_per_epoch != -1 and (step + 1) % num_protein_seq_steps_per_epoch == 0:
                cur_protein_seq_epoch += 1
                if isinstance(protein_seq_dataloader.sampler, DistributedSampler):
                    protein_seq_dataloader.sampler.set_epoch(cur_protein_seq_epoch)
                elif isinstance(protein_seq_dataloader.dataset, IterableDatasetShard):
                    protein_seq_dataloader.dataset.set_epoch(cur_protein_seq_epoch)
                protein_seq_iter = iter(protein_seq_dataloader)

            if num_protein_go_steps_per_epoch != -1 and (step + 1) % num_protein_go_steps_per_epoch == 0:
                cur_protein_go_epoch += 1
                if isinstance(protein_go_dataloader.sampler, DistributedSampler):
                    protein_go_dataloader.sampler.set_epoch(cur_protein_go_epoch)
                elif isinstance(protein_go_dataloader.dataset, IterableDatasetShard):
                    protein_go_dataloader.dataset.set_epoch(cur_protein_go_epoch)
                protein_go_iter = iter(protein_go_dataloader)

            if num_go_go_steps_per_epoch != -1 and (step + 1) % num_go_go_steps_per_epoch == 0:
                cur_go_go_epoch += 1
                if isinstance(go_go_dataloader.sampler, DistributedSampler):
                    go_go_dataloader.sampler.set_epoch(cur_go_go_epoch)
                elif isinstance(go_go_dataloader.dataset, IterableDatasetShard):
                    go_go_dataloader.dataset.set_epoch(cur_go_go_epoch)
                go_go_iter = iter(go_go_dataloader)

            protein_seq_inputs = None
            protein_go_inputs = None
            go_go_inputs = None

            if protein_seq_iter:
                protein_seq_inputs = protein_seq_iter.next()
            
            if protein_go_iter:
                protein_go_inputs = protein_go_iter.next()

            if go_go_iter:
                go_go_inputs = go_go_iter.next()

            if (
                ((step + 1) % args.gradient_accumulation_steps != 0)
                and args.local_rank != -1
                and args._no_sync_in_gradient_accumulation
            ):
                # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                with model.no_sync():
                    loss, all_loss = self.training_step(model, protein_seq_inputs, protein_go_inputs, go_go_inputs)
                    tr_loss += loss
            else:
                loss, all_loss = self.training_step(model, protein_seq_inputs, protein_go_inputs, go_go_inputs)
                tr_loss += loss

            # record loss.
            if args.local_rank == -1 or args.local_rank == 0:
                all_loss['global_step'] = step
                all_loss['learning_rate'] = self.get_learning_rate()
                all_loss = dict(all_loss)
                print(all_loss)
                self.loss_recorder.append(all_loss)

            # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
            if self.deepspeed:
                self.deepspeed.step()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                # Gradient clipping
                if args.max_grad_norm is not None and args.max_grad_norm > 0 and not self.deepspeed:
                    # deepspeed does its own clipping

                    if self.use_amp:
                        # AMP: gradients need unscaling
                        self.scaler.unscale_(self.optimizer)

                    if hasattr(self.optimizer, "clip_grad_norm"):
                        # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                        self.optimizer.clip_grad_norm(args.max_grad_norm)
                    elif hasattr(model, "clip_grad_norm_"):
                        # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                        model.clip_grad_norm_(args.max_grad_norm)
                    else:
                        # Revert to normal clipping otherwise, handling Apex or full precision
                        nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                
                # Optimizer step
                optimizer_was_run = True
                if self.deepspeed:
                    pass  # called outside the loop
                elif self.use_amp:
                    scale_before = self.scaler.get_scale()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    scale_after = self.scaler.get_scale()
                    optimizer_was_run = scale_before <= scale_after
                else:
                    self.optimizer.step()

                if optimizer_was_run and not self.deepspeed:
                    self.lr_scheduler.step()
                model.zero_grad()

            self.state.global_step += 1

            if (step+1) % 30000 == 0:
                self._save_checkpoint()
        
        logger.info("\n\nTraining completed.")
        self.is_in_train = False
        self._save_checkpoint()

    def get_learning_rate(self):
        if self.deepspeed:
            # with deepspeed's fp16 and dynamic loss scale enabled the optimizer/scheduler steps may
            # not run for the first few dozen steps while loss scale is too large, and thus during
            # that time `get_last_lr` will fail if called during that warm up stage, so work around it:
            try:
                last_lr = self.lr_scheduler.get_last_lr()
            except AssertionError as e:
                if "need to call step" in str(e):
                    logger.warning("tried to get lr value before scheduler/optimizer started stepping, returning lr=0")
                    last_lr = 0
                else:
                    raise
        else:
            last_lr = (
                # backward compatibility for pytorch schedulers
                self.lr_scheduler.get_last_lr()
                if version.parse(torch.__version__) >= version.parse("1.4")
                else self.lr_scheduler.get_lr()
            )
        return last_lr

    def _save_checkpoint(self):
        checkpoint_folder = f"checkpoint-{self.state.global_step}"

        output_dir = os.path.join(self.args.output_dir, checkpoint_folder)
        self._save(output_dir)
        if self.deepspeed:
            self.deepspeed.save_checkpoint(output_dir)

        # save loss traces.
        with open(os.path.join(output_dir, 'loss_trace.json'), 'w', encoding='utf-8') as handle:
            handle.write(json.dumps(self.loss_recorder, indent=2, ensure_ascii=False))

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        self.model.save_pretrained(output_dir, state_dict=state_dict)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]], inputs_type: str) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Prepare :obj:`inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state and aligning them to input format of `OntoProteinKELoss` and `OntoProteinMLMLoss`.

        Override `transformers.Trainer._prepare_inputs` method.

        Args:
            inputs: inputs to prepare
            inputs_type: the type of inputs, which could be choosed on {`protein_seq`, `protein_go`, `go_go`}
        """
        def to_device(inputs: Dict[str, torch.Tensor]):
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    kwargs = dict(device=self.args.device)
                    if self.deepspeed and inputs[k].dtype != torch.int64:
                        # NLP models inputs are int64 and those get adjusted to the right dtype of the
                        # embedding. Other models such as wav2vec2's inputs are already float and thus
                        # may need special handling to match the dtypes of the model
                        kwargs.update(dict(dtype=self.args.hf_deepspeed_config.dtype()))

                    inputs[k] = v.to(**kwargs)
            return inputs

        if inputs_type == 'protein_seq':
            inputs = to_device(inputs)
            return inputs
        elif inputs_type == 'protein_go' or inputs_type == 'go_go':
            postive_inputs = inputs['postive']
            negative_inputs = inputs['negative']
            postive_inputs = to_device(postive_inputs)
            negative_inputs = to_device(negative_inputs)
            return {
                'postive': postive_inputs,
                'negative': negative_inputs
            }
        else:
            raise ValueError("only support `protein_seq`, `protein_go` and `go_go`.")
        

    def training_step(
        self, 
        model: nn.Module, 
        protein_seq_inputs: Dict[str, Union[torch.Tensor, Any]] = None,
        protein_go_inputs: Dict[str, Union[torch.Tensor, Any]] = None,
        go_go_inputs: Dict[str, Union[torch.Tensor, Any]] = None
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Args:
            model: The model to train.
            protein_seq_inputs: Inputs for MLM.
            protein_go_inputs: Inputs for KE of Protein-Go.
            go_go_inputs: Inputs for KE of Go-Go.
        """
        model.train()
        protein_seq_inputs = self._prepare_inputs(protein_seq_inputs, inputs_type='protein_seq') if protein_seq_inputs else None
        protein_go_inputs = self._prepare_inputs(protein_go_inputs, inputs_type='protein_go') if protein_go_inputs else None
        go_go_inputs = self._prepare_inputs(go_go_inputs, inputs_type='go_go') if go_go_inputs else None

        if self.use_amp:
            with autocast():
                loss, all_loss = self.compute_loss(model, protein_seq_inputs=protein_seq_inputs, protein_go_inputs=protein_go_inputs, go_go_inputs=go_go_inputs)
        else:
            loss, all_loss = self.compute_loss(model, protein_seq_inputs=protein_seq_inputs, protein_go_inputs=protein_go_inputs, go_go_inputs=go_go_inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1:

            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            if not self.deepspeed:
                loss = loss / self.args.gradient_accumulation_steps

        if self.use_amp:
            self.scaler.scale(loss).backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach(), all_loss

    def compute_loss(
        self, 
        model: OntoProteinPreTrainedModel,
        protein_seq_inputs: Dict[str, Union[torch.Tensor, Any]] = None,
        protein_go_inputs: Dict[str, Union[torch.Tensor, Any]] = None,
        go_go_inputs: Dict[str, Union[torch.Tensor, Any]] = None,
    ):
        """
        Override `transformers.Trainer.compute_loss`.

        If `protein_seq_inputs` is not `None`, it will compute MLM loss.
        If `protein_go_inputs` is not `None`, It will compute KE loss on Protein-Go inputs.
        If `protein_go_inputs` is not `None`, It will compute KE loss on Go-Go inputs.
        Total loss = MLM loss + KE loss (Protein-Go) + KE loss (Go-Go) 
        """
        total_loss = torch.tensor(0.0).to(self.args.device)
        
        all_loss = collections.defaultdict(float)

        if protein_seq_inputs:
            mlm_loss = self.mlm_loss_fn(model=model, **protein_seq_inputs)
            total_loss += mlm_loss
            all_loss['mlm'] = mlm_loss.item()

        if protein_go_inputs:
            assert ('postive' in protein_go_inputs) & ('negative' in protein_go_inputs), 'Inputs need contain `postive` and `negative` keys.'
            
            postive_protein_go_inputs = protein_go_inputs['postive']
            negative_protein_go_inputs = protein_go_inputs['negative']

            ke_protein_go_postive_loss, head_relation_embed = self.ke_loss_fn(model=model, triplet_type='protein-go', is_neg=False, use_desc=self.args.use_desc, optimize_memory=False, global_step=self.state.global_step, **postive_protein_go_inputs)
            head_relation_embed = head_relation_embed if self.args.optimize_memory else None
            ke_protein_go_negative_loss, _ = self.ke_loss_fn(model=model, triplet_type='protein-go', is_neg=True, use_desc=self.args.use_desc, cache_head_relation_embed=head_relation_embed, optimize_memory=self.args.optimize_memory, **negative_protein_go_inputs)

            ke_protein_go_loss = ke_protein_go_postive_loss + ke_protein_go_negative_loss
            total_loss += ke_protein_go_loss
            all_loss['protein_go_ke'] = ke_protein_go_loss.item()

        if go_go_inputs:
            assert ('postive' in go_go_inputs) & ('negative' in go_go_inputs), 'Inputs need contain `postive` and `negative` keys.'
            
            postive_go_go_inputs = go_go_inputs['postive']
            negative_go_go_inputs = go_go_inputs['negative']

            ke_go_go_postive_loss, _ = self.ke_loss_fn(model=model, triplet_type='go-go', is_neg=False, use_desc=self.args.use_desc, **postive_go_go_inputs)
            ke_go_go_negative_loss, _ = self.ke_loss_fn(model=model, triplet_type='go-go', is_neg=True, use_desc=self.args.use_desc, **negative_go_go_inputs)

            ke_go_go_loss = ke_go_go_postive_loss + ke_go_go_negative_loss
            total_loss += ke_go_go_loss
            all_loss['go_go_ke'] = ke_go_go_loss.item()
        
        return total_loss, all_loss

    def num_examples(self, dataloader: DataLoader) -> int:
        num_examples = 0
        if dataloader:
            num_examples = len(dataloader.dataset)
        return num_examples

    def create_optimizer(self):
        """
        Setup the optimizer.

        Note: It is override from `transformers.Trainer.create_optimizer` for dynamically setting learning rate on different
        parameters.
        """
        if self.optimizer is None:
            decay_parameters = get_parameter_names(self.model, [nn.LayerNorm])
            decay_parameters = [name for name in decay_parameters if "bias" not in name]

            # -------------------- override ----------------------
            ke_parameters = get_parameter_names(self.model, [BertForMaskedLM])
            lm_parameters = get_parameter_names(self.model, [OntoModel])
            # cls_pooler_parameters = [n for n in lm_parameters if 'protein_lm.pooler' in n]
            # lm_parameters = list(set(lm_parameters) - set(cls_pooler_parameters))
            lm_decay_parameters = list(set(decay_parameters) & set(lm_parameters))
            lm_no_decay_parameters = list(set(lm_parameters) - set(decay_parameters))

            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if n in lm_decay_parameters],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.lm_learning_rate
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if n in lm_no_decay_parameters],
                    "weight_decay": 0.0,
                    'lr': self.args.lm_learning_rate
                },
                # {
                #     "params": [p for n, p in self.model.named_parameters() if n in cls_pooler_parameters],
                #     "weight_decay": self.args.weight_decay,
                #     'lr': 2e-5
                # },
                {
                    "params": [p for n, p in self.model.named_parameters() if n in ke_parameters],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.ke_learning_rate
                }
            ]
            # -----------------------------------------------------
            optimizer_cls = Adafactor if self.args.adafactor else AdamW
            if self.args.adafactor:
                optimizer_cls = Adafactor
                optimizer_kwargs = {"scale_parameter": False, "relative_step": False}
            else:
                optimizer_cls = AdamW
                optimizer_kwargs = {
                    "betas": (self.args.adam_beta1, self.args.adam_beta2),
                    "eps": self.args.adam_epsilon,
                }
            # optimizer_kwargs["lr"] = self.args.learning_rate
            # TODO: default choose `sharded_ddp` == `zero_dp_2`
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

    def create_scheduler(self, num_training_steps: int):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up before this method is called.

        Note: It is overrided from `transformer.Trainer.create_scheduler`.

        Args:
            num_training_steps (int): The number of training steps to do.
        """
        if self.lr_scheduler is None:
            # scale `num_training_steps`
            if self.args.deepspeed:
                num_training_steps = num_training_steps // self.args.gradient_accumulation_steps + int(
                    num_training_steps % self.args.gradient_accumulation_steps > 0
                )

            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                self.optimizer,
                num_lm_warmup_steps=self.args.get_lm_warmup_steps(num_training_steps),
                num_ke_warmup_steps=self.args.get_ke_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
            )

    def _get_train_sampler(self) -> Tuple:
        train_protein_seq_sampler = None
        train_protein_go_sampler = None
        train_go_go_sampler = None

        if isinstance(self.protein_seq_dataset, collections.abc.Sized):
            generator = None
            if self.args.world_size <= 1 and _is_torch_generator_available:
                generator = torch.Generator()
                generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
            
            if self.args.world_size <= 1:
                if _is_torch_generator_available:
                    train_protein_seq_sampler = RandomSampler(self.protein_seq_dataset, generator=generator)
                train_protein_seq_sampler = RandomSampler(self.protein_seq_dataset)
            else:
                train_protein_seq_sampler = DistributedSampler(
                    self.protein_seq_dataset,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    seed=self.args.seed,
                )
        
        if isinstance(self.protein_go_dataset, collections.abc.Sized):
            generator = None
            if self.args.world_size <= 1 and _is_torch_generator_available:
                generator = torch.Generator()
                generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
            
            if self.args.world_size <= 1:
                if _is_torch_generator_available:
                    train_protein_go_sampler = RandomSampler(self.protein_go_dataset, generator=generator)
                train_protein_go_sampler = RandomSampler(self.protein_go_dataset)
            else:
                train_protein_go_sampler = DistributedSampler(
                    self.protein_go_dataset,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    seed=self.args.seed,
                )

        if isinstance(self.go_go_dataset, collections.abc.Sized):
            generator = None
            if self.args.world_size <= 1 and _is_torch_generator_available:
                generator = torch.Generator()
                generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
            
            if self.args.world_size <= 1:
                if _is_torch_generator_available:
                    train_go_go_sampler = RandomSampler(self.go_go_dataset, generator=generator)
                train_go_go_sampler = RandomSampler(self.go_go_dataset)
            else:
                train_go_go_sampler = DistributedSampler(
                    self.go_go_dataset,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    seed=self.args.seed,
                )

        return train_protein_seq_sampler, train_protein_go_sampler, train_go_go_sampler

    def get_train_dataloader(self) -> Tuple:
        protein_seq_dataloader = None
        protein_go_dataloader = None
        go_go_dataloader = None

        protein_seq_sampler, protein_go_sampler, go_go_sampler = self._get_train_sampler()

        if self.protein_seq_dataset:
            protein_seq_dataloader = DataLoader(
                dataset=self.protein_seq_dataset,
                batch_size=self.args.train_protein_seq_batch_size,
                collate_fn=self.protein_seq_data_collator,
                pin_memory=self.args.dataloader_pin_memory,
                drop_last=self.args.dataloader_drop_last,
                sampler=protein_seq_sampler,
            )

        if self.protein_go_dataset:
            protein_go_dataloader = DataLoader(
                dataset=self.protein_go_dataset,
                batch_size=self.args.train_protein_go_batch_size,
                collate_fn=self.protein_go_data_collator,
                num_workers=self.args.dataloader_protein_go_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
                drop_last=self.args.dataloader_drop_last,
                sampler=protein_go_sampler,
            )

        if self.go_go_dataset:
            go_go_dataloader = DataLoader(
                dataset=self.go_go_dataset,
                batch_size=self.args.train_go_go_batch_size,
                collate_fn=self.go_go_data_collator,
                num_workers=self.args.dataloader_go_go_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
                drop_last=self.args.dataloader_drop_last,
                sampler=go_go_sampler,
            )

        return protein_seq_dataloader, protein_go_dataloader, go_go_dataloader
