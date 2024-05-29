corr_predictions = $(addprefix data/physics_predictions/large_scale_new_setup_time_,$(time_indices)) \
	$(addprefix data/physics_predictions/small_scale_new_setup_time_,$(time_indices)) \
	$(addprefix data/predictions/dnn_new_setup_time_,$(time_indices))


data/corr_coeffs.txt: $(corr_predictions) | data data/physics_predictions data/predictions .venv
	$(PYTHON) fireuq/mfmc/correlation_coeffs.py \
		--predictions=data/physics_predictions/large_scale_new_setup \
		--predictions=data/physics_predictions/small_scale_new_setup \
		--predictions=data/predictions/dnn_new_setup\
		$(addprefix --times=,$(time_indices)) \
		--corr_outfile=data/corr_coeffs.txt

corr_filtered_predictions = $(addprefix data/physics_predictions/large_scale_filtered_setup_time_,$(time_indices)) \
	$(addprefix data/physics_predictions/small_scale_filtered_setup_time_,$(time_indices)) \
	$(addprefix data/predictions/dnn_filtered_setup_time_,$(time_indices))

data/corr_coeffs_filtered.txt: $(corr_filtered_predictions) | data data/physics_predictions data/predictions .venv
	$(PYTHON) fireuq/mfmc/correlation_coeffs.py \
		--predictions=data/physics_predictions/large_scale_filtered_setup \
		--predictions=data/physics_predictions/small_scale_filtered_setup \
		--predictions=data/predictions/dnn_filtered_setup\
		$(addprefix --times=,$(time_indices)) \
		--corr_outfile=data/corr_coeffs_filtered.txt
