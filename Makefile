all: mfmc_data

time_indices = $(shell seq 1 11)

include python.mk
include download.mk
include mfmcs.mk
include correlations.mk

.PHONY: all download train_dnns dnn_predictions mfmc_data

DIRS = data data/dnns data/predictions data/mfmc


$(DIRS): %:
	mkdir -p ${*}

dnns = $(addprefix data/dnns/dnn_bytes_time_,$(time_indices))

train_dnns: $(dnns)

$(dnns): data/dnns/dnn_bytes_time_%: data/dnn_train_data | data/dnns
	$(PYTHON) fireuq/dnn/train_dnn.py \
		--train_data=data/dnn_train_data/time_${*}\
		--outdir=data/dnns/ \
		--outname=time_${*}


fn_eval_dnn = $(PYTHON) fireuq/dnn/evaluate_dnn.py \
		--outdir=data/dnns/ \
		--outname=time_$(1) \
		--input_data=data/$(2).npy \
		--eval_outname=$(3)_time_$(1) \
		--eval_outdir=data/predictions/


# Define a macro to evaluate the DNNs in the different setups
# $(1) is the time index
# $(2) is the setup
define fn_dnn_eval
dnn_$(2)_$(1): data/$(2).npy ./fireuq/dnn/evaluate_dnn.py data/dnns/dnn_bytes_time_$(1)
	$(call fn_eval_dnn,$(1),$(2),dnn_$(2))
endef

setups = new_setup new_setup_many filtered filtered_many

$(foreach setup,$(setups),$(foreach time_id,$(time_indices),$(eval $(call fn_dnn_eval,$(setup),$(time_id)))))

fn_dnn_prediction=data/predictions/dnn_$(1)_time_$(2)

dnn_predictions=$(foreach setup,$(setups),$(addprefix $(call fn_dnn_prediction,$(setup),),$(time_indices)))





