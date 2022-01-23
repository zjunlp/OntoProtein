import math
import logging
from typing import Optional, Dict
from tqdm import trange, tqdm
import torch
from torch.utils.data import DataLoader, Dataset
import transformers
from transformers.trainer_utils import TrainOutput
from transformers import get_linear_schedule_with_warmup, AdamW


logger = logging.getLogger(__name__)


def default_dev_objective(metrics):
    """
    Objective used for picking the best model on development sets
    """
    # TODO: complement remain task' dev objective
    if "eval_acc" in metrics:
        return metrics["eval_acc"]
 
    raise Exception("No metric founded for {}".format(metrics))


class Trainer(transformers.Trainer):

    def train(self, dev_objective=None, model_path=None):
        args = self.args
        self.objective = -float("inf")
        self.dev_objective = dev_objective if dev_objective is not None else default_dev_objective

        train_dataloader = self.get_train_dataloader()
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps
        num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)

        if args.max_steps > 0:
            max_steps = args.max_steps
            num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                max_steps % num_update_steps_per_epoch > 0
            )
        else:
            max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
            num_train_epochs = math.ceil(args.num_train_epochs)

        self.create_optimizer_and_scheduler(num_training_steps=max_steps)
        optimizer = self.optimizer
        scheduler = self.lr_scheduler

        model = self.model

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", self.num_examples(train_dataloader))
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per device = %d", args.per_device_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", max_steps)

        self.global_step = 0
        self.epoch = 0
        # epoch_trained = 0

        tr_loss = torch.tensor(0.0).to(self.args.device)
        logging_loss_scalar = 0
        model.zero_grad()
        for epoch in range(num_train_epochs):
            epoch_iterator = tqdm(train_dataloader, desc='Iteration')

            for step, inputs in enumerate(epoch_iterator):
                tr_loss += self.training_step(model, inputs)

                if (step + 1) % args.gradient_accumulation_steps == 0 or (
                    len(epoch_iterator) <= args.gradient_accumulation_steps
                    and (step + 1) == len(epoch_iterator)
                ):
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
                    optimizer.step()

                    scheduler.step()
                    model.zero_grad()
                    self.global_step += 1
                    self.epoch = epoch + (step + 1) / len(epoch_iterator)

                    if (self.args.logging_steps > 0 and self.global_step % self.args.logging_steps == 0) or (
                        self.global_step == 1 and self.args.logging_first_step
                    ):
                        logs = {}
                        tr_loss_scalar = tr_loss.item()
                        logs['loss'] = (tr_loss_scalar - logging_loss_scalar) / self.args.logging_steps
                        logs['norm'] = norm.item()
                        logs['learning_rate'] = scheduler.get_last_lr()[0]
                        logging_loss_scalar = tr_loss_scalar
                        
                        self.log(logs)

                    if self.args.evaluate_during_training and self.global_step % self.args.eval_steps == 0:
                        output = self.evaluate()
                        metrics = output.metrics
                        objective = self.dev_objective(metrics)
                        if objective > self.objective:
                            logger.info("Best dev result: {}".format(objective))
                            self.objective = objective
                            self.save_model(self.args.output_dir)

                if self.args.max_steps > 0 and self.global_step > self.args.max_steps:
                    epoch_iterator.close()
                    break

            if self.args.max_steps > 0 and self.global_step > self.args.max_steps:
                # train_iterator.close()
                break

        return TrainOutput(self.global_step, tr_loss / self.global_step, {'metric': self.objective})


    def evaluate(self, eval_dataset: Optional[Dataset] = None) -> Dict[str, float]:
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        output = self.prediction_loop(eval_dataloader, description="Evaluation")

        self.log(output.metrics)
        
        return output