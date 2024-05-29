default: all

time_indices = $(shell seq 1 11)

include python.mk
include download.mk
include mfmcs.mk
include correlations.mk


DIRS = data data/dnns data/predictions data/mfmc


$(DIRS): %:
	mkdir -p ${*}

dnns = $(addprefix data/dnns/dnn_bytes_time_,$(time_indices))

train_dnns: $(dnns)

$(dnns): data/dnns/dnn_bytes_time_%: data/dnn_train_data | data/dnns .venv
	$(PYTHON) fireuq/dnn/train_dnn.py \
		--train_data=data/dnn_train_data/time_${*}\
		--outdir=data/dnns/ \
		--outname=time_${*}


fn_eval_dnn = $(PYTHON) fireuq/dnn/evaluate_dnn.py \
		--outdir=data/dnns/ \
		--outname=time_$(1) \
		--input_data=data/simulation_setups/$(2).npy \
		--eval_outname=$(3)_time_$(1) \
		--eval_outdir=data/predictions/


# Define a macro to evaluate the DNNs in the different setups
# $(1) is the time index
# $(2) is the setup
define fn_dnn_eval
data/predictions/dnn_$(2)_time_$(1): data/simulation_setups/$(2).npy ./fireuq/dnn/evaluate_dnn.py data/dnns/dnn_bytes_time_$(1) | .venv data/predictions
	$(call fn_eval_dnn,$(1),$(2),dnn_$(2))
endef

setups = new_setup new_setup_many filtered_setup filtered_setup_many

$(foreach setup,$(setups),$(foreach time_id,$(time_indices),$(eval $(call fn_dnn_eval,$(time_id),$(setup)))))

all: data/corr_coeffs.txt data/corr_coeffs_filtered.txt mfmc_estimates
