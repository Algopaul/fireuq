## fireuq: Multifidelity Monte Carlo estimation of wildfires

Repository for performing MFMC for burn area prediction using [swirl_lm](github.com/googleresearch/swirl_lm).

Requires installation of `python3.11` or above

## Example usage

Run `make all` to regenerate all data tables that are presented in #TODO, add link to preprint once uploaded

To only generate parts of the data (e.g. the correlation table that generates Figure 5a), run `make data/corr_coeffs.txt`. As another example, to generate Figure 6a run `make data/mfmc/dnn_filtered_setup_10_mfmc_data.txt`

During the first run (or when the data changes), the DNNs will be (re-)trained. After that, the DNNs will be loaded from the `data/dnns` directory.

