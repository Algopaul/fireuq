import jax
import numpy as np
import jax.numpy as jnp
import flax.linen as nn
from flax.serialization import to_bytes
from absl import flags, app
import fireuq.dnn.dnn_optimize as dnopt

DNN_LAYERS = flags.DEFINE_multi_integer('dnn_layers', [5, 5, 5],
                                        'The layers of the DNN')
KEY_SEED = flags.DEFINE_integer('key_seed', 0, 'The DNN seed key')
TRAIN_DATA_FILE = flags.DEFINE_string('train_data', '',
                                      'The file containing the train data')
TRAIN_TEST_SPLIT_IDX = flags.DEFINE_integer(
    'train_test_split_idx', 900,
    'Where to split data into train and validation')
OUTNAME = flags.DEFINE_string('outname', 'desc', 'The output data name.')
OUTDIR = flags.DEFINE_string('outdir', '', 'The output directory')


class BaseMLP(nn.Module):

  @nn.compact
  def __call__(self, x) -> jax.Array:
    layers = [3, *DNN_LAYERS.value, 1]
    for layer in layers:
      x = nn.Dense(layer)(x)
      x = nn.selu(x)
    return x


def load_data(datafile=None):
  if datafile is None:
    datafile = TRAIN_DATA_FILE.value
  data = np.loadtxt(datafile, skiprows=1)
  x = data[:, :3]
  y = data[:, 3]
  return x, y


def normalize_data(x, mean_val=None, std_val=None):
  if mean_val is None:
    mean_val = np.mean(x, axis=0)
  if std_val is None:
    std_val = np.std(x, axis=0)
  return (x - mean_val) / std_val, mean_val, std_val


def denormalize_data(x, mean_val, std_val):
  return std_val * x + mean_val


def get_dnn():
  dnn = BaseMLP()
  sample = jnp.ones(3)
  params = dnn.init(jax.random.key(KEY_SEED.value), sample)
  return dnn, params


def main(_):
  x, y = load_data()
  x, x_mean, x_std = normalize_data(x)
  y, y_mean, y_std = normalize_data(y)
  dnn, init_params = get_dnn()
  tt_split = TRAIN_TEST_SPLIT_IDX.value
  x_train, x_test = x[:tt_split], x[tt_split:]
  y_train, y_test = y[:tt_split], y[tt_split:]

  @jax.jit
  def mse(params, x_batched, y_batched):
    # Define the squared loss for a single pair (x,y)
    def squared_error(x, y):
      pred = dnn.apply(params, x)
      return jnp.inner(y - pred, y - pred) / 2.0

    # Vectorize the previous to compute the average of the loss on all samples.
    return jnp.mean(jax.vmap(squared_error)(x_batched, y_batched), axis=0)

  mse(init_params, x_train, y_train)

  final_params, _ = dnopt.fit_dnn(init_params, (x_train, y_train), mse,
                                  (x_test, y_test))
  A = dnopt.collect_train_log()
  np.savetxt(OUTDIR.value + f'loss_log_{OUTNAME.value}.txt', A.T, fmt='%.4e')
  bb = to_bytes(final_params)
  with open(OUTDIR.value + 'dnn_bytes_' + OUTNAME.value, 'wb') as file:
    file.write(bb)
  np.savetxt(OUTDIR.value + 'dnn_means_' + OUTNAME.value,
             np.array([*x_mean, *x_std, y_mean, y_std]))
  pass


if __name__ == "__main__":
  app.run(main)
