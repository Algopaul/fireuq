all: mfmc_data

.PHONY: all download train_dnns dnn_predictions mfmc_data

DIRS = data data/dnns data/predictions data/mfmc

.venv:
	python3.11 -m venv .venv

install: pyproject.toml | .venv
	.venv/bin/pip install -e .

PYTHON = .venv/bin/python

$(DIRS): %:
	mkdir -p ${*}

data/dnn_train_data: | data
	gsutil -m cp -r gs://tubbs-scale-fire-simulations/dnn_train_data data/

time_ids = $(shell seq 1 11)
dnns = $(addprefix data/dnns/dnn_bytes_time_,$(time_ids))

train_dnns: $(dnns)

$(dnns): data/dnns/dnn_bytes_time_%: | data/dnns
	$(PYTHON) fireuq/dnn/train_dnn.py \
		--train_data=data/dnn_train_data/time_${*}\
		--outdir=data/dnns/ \
		--outname=time_${*}

prediction = $(foreach setup,$(2),$(addprefix data/predictions/$(1)_$(setup)_time_,$(time_ids)))
large_scale_predictions = $(call prediction,large_scale,new_setup filtered_setup)
small_scale_predictions = $(call prediction,small_scale,new_setup filtered_setup new_setup_many)

downloadable = $(large_scale_predictions) $(small_scale_predictions) $(small_scale_many_predictions)

$(downloadable): data/predictions/%:
	gsutil -m cp -r gs://tubbs-scale-fire-simulations/predictions/${*} data/predictions/${*}

download: $(downloadable)

data/filtered.npy: %:
	gsutil -m cp -r gs://tubbs-scale-fire-simulations/filtered_setup .
	$(PYTHON) -c "import numpy as np; A=np.loadtxt('filtered_setup',skiprows=1); np.save('${*}', A)"
	rm filtered_setup

data/filtered_many.npy: %:
	gsutil -m cp -r gs://tubbs-scale-fire-simulations/uq_filtered_500000_1.npy ${*}

data/new_setup.npy: %:
	gsutil -m cp -r gs://tubbs-scale-fire-simulations/uq_samples.npy ${*}

data/new_setup_many.npy: %:
	gsutil -m cp -r gs://tubbs-scale-fire-simulations/new_setup_500000_1.npy ${*}


dnn_predictions_full_range = $(call prediction,dnn,new_setup)
dnn_predictions_full_range_many = $(call prediction,dnn,new_setup_many)
dnn_predictions_filtered_range = $(call prediction,dnn,filtered_setup)
dnn_predictions_filtered_range_many = $(call prediction,dnn,filtered_setup_many)

EVAL_DNN = $(PYTHON) fireuq/dnn/evaluate_dnn.py \
		--outdir=data/dnns/ \
		--outname=time_$(1) \
		--input_data=data/$(2).npy \
		--eval_outname=$(3)_time_$(1) \
		--eval_outdir=data/predictions/

$(dnn_predictions_filtered_range): data/predictions/dnn_filtered_setup_time_%: data/filtered.npy ./fireuq/dnn/evaluate_dnn.py data/dnns/dnn_bytes_time_%
	$(call EVAL_DNN,${*},filtered,dnn_filtered_setup)

$(dnn_predictions_filtered_range_many): data/predictions/dnn_filtered_setup_many_time_%: data/filtered_many.npy ./fireuq/dnn/evaluate_dnn.py data/dnns/dnn_bytes_time_%
	$(call EVAL_DNN,${*},filtered_many,dnn_filtered_setup_many)

$(dnn_predictions_full_range): data/predictions/dnn_new_setup_time_%: data/new_setup.npy ./fireuq/dnn/evaluate_dnn.py data/dnns/dnn_bytes_time_%
	$(call EVAL_DNN,${*},new_setup,dnn_new_setup)

$(dnn_predictions_full_range_many): data/predictions/dnn_new_setup_many_time_%: data/new_setup_many.npy ./fireuq/dnn/evaluate_dnn.py data/dnns/dnn_bytes_time_%
	$(call EVAL_DNN,${*},new_setup_many,dnn_new_setup_many)

dnn_predictions: $(dnn_predictions_filtered_range_many) $(dnn_predictions_full_range) $(dnn_predictions_full_range_many) $(dnn_predictions_filtered_range)


mfmc = $(addsuffix _mfmc_data.txt,$(addprefix data/mfmc/$(1)_,$(time_ids)))

mfmcs = $(call mfmc,dnn_new_setup)
mfmcs_sub = $(call mfmc,sub_dnn_new_setup)
mfmcs_filtered = $(call mfmc,dnn_filtered_setup)
mfmcs_filtered_sub = $(call mfmc,sub_dnn_filtered_setup)

MFMC = $(PYTHON) fireuq/mfmc/mfmc_analysis.py\
		--cost_large=1536.0 \
		--cost_small=1.0e-5\
		--sample_file_large=./data/predictions/$(1)\
		--sample_file_small=./data/predictions/$(2)\
		--sample_file_small_many=./data/predictions/$(3)\
		--outfile=./data/mfmc/$(4)

interest_budgets = 5000.0 10000.0 15000.0 20000.0 25000.0 30000.0

MFMC_sub = $(PYTHON) fireuq/mfmc/mfmc_analysis.py\
		--cost_large=1536.0 \
		--cost_small=1.0e-5\
		--sample_file_large=./data/predictions/$(1)\
		--sample_file_small=./data/predictions/$(2)\
		--sample_file_small_many=./data/predictions/$(3)\
		--outfile=./data/mfmc/sub_$(4)\
		$(addprefix --budgets ,$(interest_budgets))


$(mfmcs): data/mfmc/dnn_new_setup_%_mfmc_data.txt: ./data/predictions/large_scale_new_setup_time_% ./data/predictions/dnn_new_setup_time_% ./data/predictions/dnn_new_setup_many_time_% | data/mfmc
	$(call MFMC,large_scale_new_setup_time_${*},dnn_new_setup_time_${*},dnn_new_setup_many_time_${*},dnn_new_setup_${*})

$(mfmcs_sub): data/mfmc/sub_dnn_new_setup_%_mfmc_data.txt: ./data/predictions/large_scale_new_setup_time_% ./data/predictions/dnn_new_setup_time_% ./data/predictions/dnn_new_setup_many_time_% | data/mfmc
	$(call MFMC_sub,large_scale_new_setup_time_${*},dnn_new_setup_time_${*},dnn_new_setup_many_time_${*},dnn_new_setup_${*})

$(mfmcs_filtered): data/mfmc/dnn_filtered_setup_%_mfmc_data.txt: ./data/predictions/large_scale_filtered_setup_time_% ./data/predictions/dnn_filtered_setup_time_% ./data/predictions/dnn_filtered_setup_many_time_% | data/mfmc
	$(call MFMC,large_scale_filtered_setup_time_${*},dnn_filtered_setup_time_${*},dnn_filtered_setup_many_time_${*},dnn_filtered_setup_${*})

$(mfmcs_filtered_sub): data/mfmc/sub_dnn_filtered_setup_%_mfmc_data.txt: ./data/predictions/large_scale_filtered_setup_time_% ./data/predictions/dnn_filtered_setup_time_% ./data/predictions/dnn_filtered_setup_many_time_% | data/mfmc
	$(call MFMC_sub,large_scale_filtered_setup_time_${*},dnn_filtered_setup_time_${*},dnn_filtered_setup_many_time_${*},dnn_filtered_setup_${*})


mfmc_data: $(mfmcs) $(mfmcs_filtered) $(mfmcs_sub) $(mfmcs_filtered_sub)

corr_predictions = $(addprefix data/predictions/large_scale_new_setup_time_,$(time_ids)) \
	$(addprefix data/predictions/small_scale_new_setup_time_,$(time_ids)) \
	$(addprefix data/predictions/dnn_new_setup_time_,$(time_ids))


data/corr_coeffs.txt: $(corr_predictions) | data
	$(PYTHON) fireuq/mfmc/correlation_coeffs.py \
		--predictions=data/predictions/large_scale_new_setup \
		--predictions=data/predictions/small_scale_new_setup \
		--predictions=data/predictions/dnn_new_setup\
		$(addprefix --times=,$(time_ids)) \
		--corr_outfile=data/corr_coeffs.txt

corr_filtered_predictions = $(addprefix data/predictions/large_scale_filtered_setup_time_,$(time_ids)) \
	$(addprefix data/predictions/small_scale_filtered_setup_time_,$(time_ids)) \
	$(addprefix data/predictions/dnn_filtered_setup_time_,$(time_ids))

data/corr_coeffs_filtered.txt: $(corr_filtered_predictions) | data
	$(PYTHON) fireuq/mfmc/correlation_coeffs.py \
		--predictions=data/predictions/large_scale_filtered_setup \
		--predictions=data/predictions/small_scale_filtered_setup \
		--predictions=data/predictions/dnn_filtered_setup\
		$(addprefix --times=,$(time_ids)) \
		--corr_outfile=data/corr_coeffs_filtered.txt
