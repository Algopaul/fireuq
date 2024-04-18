import jax
import numpy as np
import jax.numpy as jnp
import flax.linen as nn
from flax.serialization import to_bytes, from_bytes
from absl import flags, app
import fireuq.dnn.dnn_optimize as dnopt
from fireuq.dnn.train_dnn import get_dnn, OUTDIR, OUTNAME, normalize_data, denormalize_data

INPUT_DATA = flags.DEFINE_string('input_data', '',
                                 'The input data for DNN inference')
EVAL_OUTNAME = flags.DEFINE_string('eval_outname', 'desc',
                                   'The output data name.')
EVAL_OUTDIR = flags.DEFINE_string('eval_outdir', '', 'The output directory')


def load_means():
  A = np.loadtxt(OUTDIR.value + 'dnn_means_' + OUTNAME.value)
  x_mean = A[:3]
  x_std = A[3:6]
  y_mean = A[6]
  y_std = A[7]
  return x_mean, x_std, y_mean, y_std


def main(_):
  dnn, params = get_dnn()
  with open(OUTDIR.value + 'dnn_bytes_' + OUTNAME.value, 'rb') as file:
    trained_bytes = file.read()
  trained_params = from_bytes(params, trained_bytes)
  f = jax.vmap(dnn.apply, in_axes=[None, 0])
  x_mean, x_std, y_mean, y_std = load_means()
  x = np.load(INPUT_DATA.value)
  x_normalized = normalize_data(x, x_mean, x_std)[0]
  y_normalized = f(trained_params, x_normalized)
  y = denormalize_data(y_normalized, y_mean, y_std)
  np.savetxt(EVAL_OUTDIR.value + EVAL_OUTNAME.value, y, header='burned area')


if __name__ == "__main__":
  app.run(main)
