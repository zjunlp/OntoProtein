from typing import Union, Optional
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from transformers.trainer_utils import SchedulerType


def get_linear_schedule_with_warmup(
    optimizer: Optimizer,
    num_training_steps: int,
    num_lm_warmup_steps: int,
    num_ke_warmup_steps: int,
    last_epoch=-1
):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Note: It is override from `transformers.optimization.get_linear_schedule_with_warmup` in view of dynamically setting of linear
    schedule on different parameter group. This method only implement two type of `lr_lambda` function respectively to the schedule
    of learning rate on KE and LM.
    """

    # LM
    def lr_lambda_for_lm(current_step: int):
        if current_step < num_lm_warmup_steps:
            return float(current_step) / float(max(1, num_lm_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_lm_warmup_steps))
        )

    # KE
    def lr_lambda_for_ke(current_step: int):
        if current_step < num_ke_warmup_steps:
            return float(current_step) / float(max(1, num_ke_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_ke_warmup_steps))
        )

    # CLS pooler
    def lr_lambda_for_pooler(current_step: int):
        if current_step < num_lm_warmup_steps:
            return float(current_step) / float(max(1, num_lm_warmup_steps))
        
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_lm_warmup_steps))
        )

    return LambdaLR(optimizer, [lr_lambda_for_lm, lr_lambda_for_lm, lr_lambda_for_ke], last_epoch)


TYPE_TO_SCHEDULER_FUNCTION = {
    SchedulerType.LINEAR: get_linear_schedule_with_warmup,
}


def get_scheduler(
    name: str,
    optimizer: Optimizer,
    num_lm_warmup_steps: Optional[int] = None,
    num_ke_warmup_steps: Optional[int] = None,
    num_training_steps: Optional[int] = None,
):
    """
    Unified API to get any scheduler from its name.

    Args:
        name (:obj:`str` or `:obj:`SchedulerType`):
            The name of the scheduler to use.
        optimizer (:obj:`torch.optim.Optimizer`):
            The optimizer that will be used during training.
        num_warmup_steps (:obj:`int`, `optional`):
            The number of warmup steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
        num_training_steps (:obj:`int`, `optional`):
            The number of training steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
    """
    name = SchedulerType(name)
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]

    # All other schedulers require `num_warmup_steps`
    if num_ke_warmup_steps is None or num_lm_warmup_steps is None:
        raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")

    # All other schedulers require `num_training_steps`
    if num_training_steps is None:
        raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")

    return schedule_func(
        optimizer, 
        num_lm_warmup_steps=num_lm_warmup_steps, 
        num_ke_warmup_steps=num_ke_warmup_steps, 
        num_training_steps=num_training_steps
    )