import torch
from transformers.deepspeed import HfTrainerDeepSpeedConfig


class OntoProteinTrainerDeepSpeedConfig(HfTrainerDeepSpeedConfig):
    """
    The ``OntoProteinTrainerDeepSpeedConfig`` object is meant to be created during ``TrainingArguments`` object creation and has
    the same lifespan as the latter.

    Note: It is rewrited from `HfTrainerDeepSpeedConfig`.
    """

    def trainer_config_process(self, args):
        train_protein_seq_batch_size = args.world_size * args.per_device_train_protein_seq_batch_size * args.gradient_accumulation_steps
        train_protein_go_batch_size = args.world_size * args.per_device_train_protein_go_batch_size * args.gradient_accumulation_steps
        train_go_go_batch_size = args.world_size * args.per_device_train_go_go_batch_size * args.gradient_accumulation_steps

        self.fill_match(
            "train_micro_protein_seq_batch_size_per_gpu", args.per_device_train_protein_seq_batch_size, "per_device_train_protein_seq_batch_size"
        )
        self.fill_match(
            "train_micro_protein_go_batch_size_per_gpu", args.per_device_train_protein_go_batch_size, "per_device_train_protein_go_batch_size"
        )
        self.fill_match(
            "train_micro_go_go_batch_size_per_gpu", args.per_device_train_go_go_batch_size, "per_device_train_go_go_batch_size"
        )
        self.fill_match("gradient_accumulation_steps", args.gradient_accumulation_steps, "gradient_accumulation_steps")
        self.fill_match("train_protein_seq_batch_size", train_protein_seq_batch_size, "train_protein_seq_batch_size (calculated)")
        self.fill_match("train_protein_go_batch_size", train_protein_go_batch_size, "train_protein_go_batch_size (calculated)")
        self.fill_match("train_go_go_batch_size", train_go_go_batch_size, "train_go_go_batch_size (calculated)")
        self.fill_match("gradient_clipping", args.max_grad_norm, "max_grad_norm")

        self.fill_match("optimizer.params.lm_lr", args.lm_learning_rate, "lm_learning_rate")
        self.fill_match("optimizer.params.ke_lr", args.ke_learning_rate, "ke_learning_rate")

        self.fill_match("optimizer.params.betas", [args.adam_beta1, args.adam_beta2], "adam_beta1+adam_beta2")
        self.fill_match("optimizer.params.eps", args.adam_epsilon, "adam_epsilon")
        self.fill_match("optimizer.params.weight_decay", args.weight_decay, "weight_decay")

        self.fill_only("scheduler.params.warmup_min_lr", 0)  # not a trainer arg
        self.fill_match("scheduler.params.warmup_max_mlm_lr", args.lm_learning_rate, "lm_learning_rate")
        # total_num_steps - will get set in trainer_config_finalize

        # fp16
        if args.fp16:
            fp16_backend = "apex" if args.fp16_backend == "apex" else "amp"
        else:
            fp16_backend = None

        # amp: similar to the pytorch native amp - it has a bunch of optional params but we won't set
        # any here unless the user did the work
        self.fill_match("fp16.enabled", fp16_backend == "amp", "fp16+fp16_backend(amp)")

        # apex: delegates amp work to apex (which needs to be available), but it cannot be used with any
        # ZeRO features
        self.fill_match("amp.enabled", fp16_backend == "apex", "fp16+fp16_backend(apex)")
        self.fill_match("amp.opt_level", args.fp16_opt_level, "fp16_opt_level")

        # only if we have an explicit fp16.enabled = False then it's fp32, if it's True or this
        # whole config section is missing then the fallback is fp16
        if self.is_false("fp16.enabled"):
            self._dtype = torch.float32
        # later there will be other dtypes besides just fp16 and fp32
        # also not quite sure what dtype should be under apex, defaulting to fp16 for now
