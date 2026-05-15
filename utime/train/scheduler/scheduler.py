import keras
import keras.ops as ops
import numpy as np
import logging
from utime.train.utils import _get_classes_or_funcs
import utime

logger = logging.getLogger(__name__)
# ============================================================
# Mutable backend-agnostic schedule
# ============================================================

class MutableSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr):
        super().__init__()
        self.current_lr = float(initial_lr)

    def __call__(self, step):
        return ops.convert_to_tensor(self.current_lr)

    def set_lr(self, lr):
        self.current_lr = float(lr)

    def get_config(self):
        return {"initial_lr": self.current_lr}

class FlexibleScheduler(keras.callbacks.Callback):
    """
    schedule_fn signature:

        fn(
            index,
            current_lr,
            logs,
            phase,
        ) -> new_lr

    phase:
        "batch"
        "epoch"
    """

    def __init__(
        self,
        schedule,
        schedule_fn,
        update_on="epoch",
    ):
        super().__init__()

        self.schedule = schedule
        self.schedule_fn = schedule_fn
        self.update_on = update_on

    def _apply_update(self, index, logs, phase):
        old_lr = self.schedule.current_lr
        new_lr = self.schedule_fn(
            index
        )

        self.schedule.set_lr(new_lr)

        logger.info(
            f"{phase:<5} "
            f"index={index:<6} "
            f"lr {old_lr:.8f} -> {new_lr:.8f}"
        )

    def on_train_batch_begin(self, batch, logs=None):
        if self.update_on != "batch":
            return

        # TRUE global training iteration
        global_step = int(self.model.optimizer.iterations)

        self._apply_update(
            index=global_step,
            logs=logs,
            phase="batch",
        )

    def on_epoch_begin(self, epoch, logs=None):
        if self.update_on != "epoch":
            return

        self._apply_update(
            index=epoch,
            logs=logs,
            phase="epoch",
        )
        
def init_scheduler(scheduler_name, lr, scheduler_kwargs):
    if scheduler_name is None:
        return {'sheduler': lr, 'scheduler_callback': None}
    scheduler_class = _get_classes_or_funcs(scheduler_name, func_modules=[utime.train.scheduler])
    if len(scheduler_class) != 1:
        return {'sheduler': lr, 'scheduler_callback': None}
    scheduler_fn = scheduler_class[0](**scheduler_kwargs)
    sheduler = MutableSchedule(initial_lr=scheduler_kwargs.get('initial_learning_rate', lr))
    scheduler_callback = FlexibleScheduler(schedule=sheduler, schedule_fn=scheduler_fn, update_on=scheduler_fn.step)
    return {'sheduler': sheduler, 'scheduler_callback': scheduler_callback}

class PolynomialDecaySchedule:
    def __init__(
        self,
        initial_learning_rate,
        decay_steps,
        end_learning_rate=1e-7,
        power=1.0,
        step="epoch",
        **kwargs
    ):
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.step = step
    
    def __call__(self, step):
        initial_learning_rate = ops.convert_to_tensor(
            self.initial_learning_rate
        )
        dtype = initial_learning_rate.dtype
        end_learning_rate = ops.cast(self.end_learning_rate, dtype)
        power = ops.cast(self.power, dtype)

        global_step_recomp = ops.cast(step, dtype)
        decay_steps_recomp = ops.cast(self.decay_steps, dtype)
        global_step_recomp = ops.minimum(
            global_step_recomp, decay_steps_recomp
        )
        p = ops.divide(global_step_recomp, decay_steps_recomp)
        return ops.add(
            ops.multiply(
                initial_learning_rate - end_learning_rate,
                ops.power(1 - p, power),
            ),
            end_learning_rate,
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            "decay_steps": self.decay_steps,
            "end_learning_rate": self.end_learning_rate,
            "power": self.power,
            "step": self.step,
        })
        return config


        