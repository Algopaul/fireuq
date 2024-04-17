all: mfmc_data

DIRS = data data/dnns data/predictions data/mfmc

$(DIRS): %:
	mkdir -p ${*}

data/dnn_train_data: | data
	gsutil -m cp -r gs://tubbs-scale-fire-simulations/dnn_train_data data/

time_ids = $(shell seq 1 11)
dnns = $(addprefix data/dnns/dnn_bytes_time_,$(time_ids))

train_dnns: $(dnns)

$(dnns): data/dnns/dnn_bytes_time_%: | data/dnns
	python fireuq/dnn/train_dnn.py \
		--train_data=data/dnn_train_data/time_${*}\
		--outdir=data/dnns/ \
		--outname=time_${*}

large_scale_predictions = $(addprefix data/predictions/large_scale_new_setup_time_,$(time_ids))
large_scale_filtered_predictions = $(addprefix data/predictions/large_scale_filtered_setup_time_,$(time_ids))
small_scale_predictions = $(addprefix data/predictions/small_scale_new_setup_time_,$(time_ids))
small_scale_many_predictions = $(addprefix data/predictions/small_scale_new_setup_many_time_,$(time_ids))

download: $(large_scale_predictions) $(small_scale_predictions) $(small_scale_many_predictions) $(large_scale_filtered_predictions)

$(large_scale_predictions) $(small_scale_predictions) $(small_scale_many_predictions) $(large_scale_filtered_predictions): data/predictions/%:
	gsutil -m cp -r gs://tubbs-scale-fire-simulations/predictions/${*} data/predictions/${*}


data/filtered.npy: %:
	gsutil -m cp -r gs://tubbs-scale-fire-simulations/filtered_setup .
	python -c "import numpy as np; A=np.loadtxt('filtered_setup',skiprows=1); np.save('${*}', A)"
	rm filtered_setup

data/filtered_many.npy: %:
	gsutil -m cp -r gs://tubbs-scale-fire-simulations/uq_filtered_500000_1.npy ${*}

data/new_setup.npy: %:
	gsutil -m cp -r gs://tubbs-scale-fire-simulations/uq_samples.npy ${*}

data/new_setup_many.npy: %:
	gsutil -m cp -r gs://tubbs-scale-fire-simulations/new_setup_500000_1.npy ${*}


dnn_predictions_full_range = $(addprefix data/predictions/dnn_new_setup_time_,$(time_ids))
dnn_predictions_full_range_many = $(addprefix data/predictions/dnn_new_setup_many_time_,$(time_ids))
dnn_predictions_filtered_range = $(addprefix data/predictions/dnn_filtered_setup_time_,$(time_ids))
dnn_predictions_filtered_range_many = $(addprefix data/predictions/dnn_filtered_setup_many_time_,$(time_ids))

EVAL_DNN = python fireuq/dnn/evaluate_dnn.py \
		--outdir=data/dnns/ \
		--outname=time_$(1) \
		--input_data=data/$(2).npy \
		--eval_outname=$(3)_time_$(1) \
		--eval_outdir=data/predictions/

$(dnn_predictions_filtered_range): data/predictions/dnn_filtered_setup_time_%: data/filtered.npy ./fireuq/dnn/evaluate_dnn.py data/dnns/dnn_bytes_time_%
	$(call EVAL_DNN,${*},filtered,dnn_filtered_setup)

$(dnn_predictions_filtered_range_many): data/predictions/dnn_filtered_setup_many_time_%: data/filtered_many.npy ./fireuq/dnn/evaluate_dnn.py
	$(call EVAL_DNN,${*},filtered_many,dnn_filtered_setup_many)

$(dnn_predictions_full_range): data/predictions/dnn_new_setup_time_%: data/new_setup.npy ./fireuq/dnn/evaluate_dnn.py
	$(call EVAL_DNN,${*},new_setup,dnn_new_setup)

$(dnn_predictions_full_range_many): data/predictions/dnn_new_setup_many_time_%: data/new_setup_many.npy ./fireuq/dnn/evaluate_dnn.py
	$(call EVAL_DNN,${*},new_setup_many,dnn_new_setup_many)

dnn_predictions: $(dnn_predictions_filtered_range_many) $(dnn_predictions_full_range) $(dnn_predictions_full_range_many) $(dnn_predictions_filtered_range)

mfmcs = $(addsuffix _mfmc_data.txt,$(addprefix data/mfmc/dnn_new_setup_,$(time_ids)))
mfmcs_filtered = $(addsuffix _mfmc_data.txt,$(addprefix data/mfmc/dnn_filtered_setup_,$(time_ids)))

$(mfmcs): data/mfmc/dnn_new_setup_%_mfmc_data.txt: ./data/predictions/large_scale_new_setup_time_% ./data/predictions/dnn_new_setup_time_% ./data/predictions/dnn_new_setup_many_time_% | data/mfmc
	python fireuq/mfmc/mfmc_analysis.py\
		--cost_large=1536.0 \
		--cost_small=1.0e-5\
		--sample_file_large=./data/predictions/large_scale_new_setup_time_${*}\
		--sample_file_small=./data/predictions/dnn_new_setup_time_${*}\
		--sample_file_small_many=./data/predictions/dnn_new_setup_many_time_${*}\
		--outfile=./data/mfmc/dnn_new_setup_${*}

$(mfmcs_filtered): data/mfmc/dnn_filtered_setup_%_mfmc_data.txt: ./data/predictions/large_scale_filtered_setup_time_% ./data/predictions/dnn_filtered_setup_time_% ./data/predictions/dnn_filtered_setup_many_time_% | data/mfmc
	python fireuq/mfmc/mfmc_analysis.py\
		--cost_large=1536.0 \
		--cost_small=1.0e-5\
		--sample_file_large=./data/predictions/large_scale_filtered_setup_time_${*}\
		--sample_file_small=./data/predictions/dnn_filtered_setup_time_${*}\
		--sample_file_small_many=./data/predictions/dnn_filtered_setup_many_time_${*}\
		--outfile=./data/mfmc/dnn_filtered_setup_${*}

mfmc_data: $(mfmcs) $(mfmcs_filtered)
