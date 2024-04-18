import optax
from functools import partial
import jax
from absl import flags, logging
import numpy as np

_SCHEDULE_LEARNING_RATE = flags.DEFINE_bool(
    'schedule_learning_rate', True, 'Whether to use a learning_rate scheduler.')
_START_LEARNING_RATE = flags.DEFINE_float('start_learning_rate', 1e-3,
                                          'The base/initial learning rate')
_END_LEARNING_RATE = flags.DEFINE_float('end_learning_rate', 1e-8,
                                        'The final learning rate')
_N_WARMUP_STEPS = flags.DEFINE_integer('n_warmup_steps', 50,
                                       'The number of warmup steps')
_N_OPTIMIZATION_STEPS = flags.DEFINE_integer(
    'n_optimization_steps', 100000, 'The number of optimization steps')
_INITIALIZATION_LOSS_TOL = flags.DEFINE_float(
    'initialization_loss_tol', 0.0,
    'Breaks the optimization loop if loss is below this value.')

TRAIN_LOSSES = []
VAL_LOSSES = []
ITERS = []

OPTIMIZER_DICT = {
    "adam": optax.adam,
    "adamw": optax.adamw,
    "adamww": partial(optax.adamw, weight_decay=1e-3),
    "amsgrad": optax.amsgrad,
    "adabelief": optax.adabelief,
}


def get_optimizer(optimizer_name,
                  learning_rate) -> optax.GradientTransformation:
  opti_f = OPTIMIZER_DICT[optimizer_name]
  return opti_f(learning_rate=learning_rate)


def standard_loss(param, x_train, dnn, y_train):
  vmapped_dnn = jax.vmap(dnn.apply, in_axes=[None, 0])
  return jax.numpy.mean(
      jax.numpy.squeeze(vmapped_dnn(param, x_train) - y_train)**2)


def fit_dnn(
    param_init,
    args,
    loss,
    val_data,
    steps=None,
    learning_rate=None,
    loss_tol=None,
    optimizer_name="adam",
):
  if loss_tol is None:
    loss_tol = _INITIALIZATION_LOSS_TOL.value
  if learning_rate is None:
    learning_rate = _START_LEARNING_RATE.value
  if steps is None:
    steps = _N_OPTIMIZATION_STEPS.value
  if _SCHEDULE_LEARNING_RATE.value:
    learning_rate = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        warmup_steps=_N_WARMUP_STEPS.value,
        peak_value=_START_LEARNING_RATE.value,
        decay_steps=_N_OPTIMIZATION_STEPS.value,
        end_value=_END_LEARNING_RATE.value)
  optimizer = get_optimizer(optimizer_name, learning_rate)
  opt_state = optimizer.init(param_init)

  @jax.jit
  def step(params, opt_state, args):
    loss_value, grads = jax.value_and_grad(loss)(params, *args)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss_value

  loss_value = 0.0

  params = param_init
  final_params = param_init
  val_loss = loss(params, *val_data)

  print('steps')
  print(steps)
  print('steps')
  for i in range(steps):
    params, opt_state, loss_value = step(params, opt_state, args)
    if i % 100 == 0:
      ITERS.append(i)
      TRAIN_LOSSES.append(loss_value)
      c_val_loss = loss(params, *val_data)
      VAL_LOSSES.append(c_val_loss)
      if c_val_loss < val_loss:
        final_params = params
        val_loss = c_val_loss
      logging.info(f'step {i}, loss: {loss_value:.5e}')
    if loss_value < loss_tol:
      break

  return final_params, loss_value


def collect_train_log():
  iters = np.stack(ITERS)
  train_losses = np.stack(TRAIN_LOSSES)
  val_losses = np.stack(VAL_LOSSES)
  A = np.stack((iters, train_losses, val_losses))
  return A
