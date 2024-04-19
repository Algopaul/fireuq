import numpy as np
from absl import flags, app

flags.DEFINE_multi_string('predictions', [], 'The predictions to compare')
flags.DEFINE_multi_integer('times', [], 'The time ids at which to compare')
flags.DEFINE_string('corr_outfile', '', 'The output file to write the results')


def remove_nans(x, y):
  idcs = np.logical_not(np.logical_or(np.isnan(x), np.isnan(y)))
  return x[idcs], y[idcs]


def variances_correlation(x, y):
  sigma_x = np.sqrt(np.var(x, ddof=1))
  sigma_y = np.sqrt(np.var(y, ddof=1))
  cov_xy = 1 / (len(x) - 1) * np.inner(x - np.mean(x), y - np.mean(y))
  rho_xy = cov_xy / (sigma_x * sigma_y)
  return sigma_x, sigma_y, rho_xy


def load_result(prediction, time):
  return np.loadtxt(f"{prediction}_time_{time}", skiprows=1)


def main(_):
  predictions = flags.FLAGS.predictions
  times = flags.FLAGS.times
  outfile = flags.FLAGS.corr_outfile
  results = []
  rhos = np.zeros((len(times), len(predictions) * (len(predictions) - 1) // 2))
  for i, prediction in enumerate(predictions):
    results_i = []
    for j, time in enumerate(times):
      results_i.append(load_result(prediction, time))
    results.append(results_i)
  print(len(results))
  rhos = []
  for i in range(len(results)):
    for j in range(i + 1, len(results)):
      rhos_ij = []
      for t in range(len(times)):
        x = results[i][t]
        y = results[j][t]
        xr, yr = remove_nans(x, y)
        rhos_ij.append(variances_correlation(xr, yr)[2])
      rhos.append(rhos_ij)
  np.savetxt(outfile, np.array([times, *rhos]).T)


if __name__ == "__main__":
  app.run(main)
