import numpy as np
from absl import flags, app
from fireuq.mfmc.correlation_coeffs import variances_correlation, remove_nans

COST_LARGE = flags.DEFINE_float(
    'cost_large', 100.0, 'The cost of large scale simulation in TPU hours')
COST_SMALL = flags.DEFINE_float(
    'cost_small', 10.0, 'The cost of surrogate evaluation in TPU hours')
SAMPLE_FILE_LARGE = flags.DEFINE_string('sample_file_large', '',
                                        'File containing large scale samples')
SAMPLE_FILE_SMALL = flags.DEFINE_string('sample_file_small', '',
                                        'File containing small scale samples')
SAMPLE_FILE_SMALL_MANY = flags.DEFINE_string(
    'sample_file_small_many', '', 'File containing many small scale samples')
OUTFILE = flags.DEFINE_string('outfile', '',
                              'File base for solving the results')
BUDGETS = flags.DEFINE_multi_float('budgets', [], 'The budgets to use')


def load_results():
  large_scale = np.loadtxt(SAMPLE_FILE_LARGE.value, skiprows=1)
  small_prev = np.loadtxt(SAMPLE_FILE_SMALL.value, skiprows=1)
  small_new = np.loadtxt(SAMPLE_FILE_SMALL_MANY.value, skiprows=1)
  return large_scale, small_prev, small_new


def mc_estimate(x, budget, budget_per_run):
  n_runs = intfloor(budget / budget_per_run)
  x = x[:n_runs]
  return mean_na_ignore(x)


def mean_na_ignore(x):
  idcs = np.logical_not(np.isnan(x))
  return np.mean(x[idcs])


def mfmc_params(x, y, budget_per_run_ls, budget_per_run_sm):
  x, y = remove_nans(x, y)
  sigma_1, sigma_21, rho12 = variances_correlation(x, y)
  alpha = rho12 * sigma_1 / sigma_21
  w1 = budget_per_run_ls
  w2 = budget_per_run_sm
  w = np.stack((w1, w2))
  r1 = np.sqrt(w1 * (1 - rho12**2) / (w1 * (1 - rho12**2)))
  r2 = np.sqrt(w1 * (rho12**2) / (w2 * (1 - rho12**2)))
  r = np.stack((r1, r2))
  return rho12, alpha, w, r


def mfmc_compute_n_runs(w, r, budget):
  n_ls = budget / np.inner(w, r)
  n_sm = n_ls * r[1]
  return intfloor(n_ls), intfloor(n_sm)


def intfloor(x):
  return np.floor(x).astype('int')


def mfmc_estimate(budget, x_ls, x_sm, x_new_sm, alpha, w, r):
  n_ls, n_sm = mfmc_compute_n_runs(w, r, budget)
  if n_ls < 1:
    n_ls = 1
    n_sm = 0
    y1m1 = mean_na_ignore(x_ls[:n_ls])
    return y1m1, n_ls, n_sm
  y1m1 = mean_na_ignore(x_ls[:n_ls])
  y2m1 = mean_na_ignore(x_sm[:n_ls])
  y2m2 = mean_na_ignore(np.hstack((x_sm[:n_ls], x_new_sm[:n_sm])))
  return y1m1 + alpha * (y2m2 - y2m1), n_ls, n_sm


def max_budget_valid(budgets, n_ls, n_sm, n_samples_ls, n_samples_sm):
  if n_ls[-1] <= n_samples_ls:
    p1 = len(budgets) - 1
  else:
    p1 = np.argmax(n_ls > n_samples_ls)
  if n_sm[-1] <= n_samples_sm:
    p2 = len(budgets) - 1
  else:
    p2 = np.argmax(n_sm > n_samples_sm)
  return budgets[np.minimum(p1, p2)]


def main(_):
  ls_results, sm_results, sm_results_new = load_results()
  ls_results, sm_results = remove_nans(ls_results, sm_results)
  bpr_large_scale = COST_LARGE.value
  bpr_small_scale = COST_SMALL.value
  if len(BUDGETS.value) == 0:
    budgets = np.linspace(bpr_large_scale, 30000, 100)
  else:
    budgets = BUDGETS.value

  mc_large = np.asarray(
      [mc_estimate(ls_results, budget, bpr_large_scale) for budget in budgets])
  mc_small = np.asarray(
      [mc_estimate(sm_results, budget, bpr_small_scale) for budget in budgets])
  sigma_1 = np.var(ls_results,ddof=1)

  # mfmc
  cor_coeff, alpha, w, r = mfmc_params(ls_results, sm_results, bpr_large_scale,
                                       bpr_small_scale)
  mfmc_args = (ls_results, sm_results, sm_results_new, alpha, w, r)
  mfmc, n_ls, n_sm = np.asarray(
      [mfmc_estimate(budget, *mfmc_args) for budget in budgets]).T
  max_budget = max_budget_valid(budgets, n_ls, n_sm, len(ls_results),
                                len(sm_results_new))
  variance_reduction_ratio = (
      np.sqrt(1 - cor_coeff**2) +
      np.sqrt(bpr_small_scale / bpr_large_scale * cor_coeff**2))**2
  mc_variance = sigma_1 / n_ls
  mfmc_variance = mc_variance * variance_reduction_ratio
  np.savetxt(
      OUTFILE.value + '_mfmc_data.txt',
      np.stack((budgets, mc_large, mc_small, mfmc, n_ls, n_sm, mc_variance,
                mfmc_variance)).T,
      delimiter='\t',
      header='budgets\tmc_large\tmc_small\tmfmc\tn_ls\tn_sm\tmc_variance\tmfmc_variance',
      comments='',
      fmt='%.6e',
  )
  np.savetxt(
      OUTFILE.value + '_test_results.txt',
      np.stack((ls_results, sm_results)).T,
      delimiter='\t',
      header='lst_results\tsm_results',
      comments='',
      fmt='%.6e',
  )
  with open(OUTFILE.value + '_info.txt', 'w') as file:
    file.write(f'{max_budget}\t')
    file.write(f'{cor_coeff}\t')
    file.write(f'{alpha}\t')
  for var, name in zip([ls_results, sm_results, sm_results_new],
                       ['ls', 'sm', 'sm_new']):
    np.savetxt(OUTFILE.value + name + '.txt', var, comments='', fmt='%.6e')


if __name__ == "__main__":
  app.run(main)
