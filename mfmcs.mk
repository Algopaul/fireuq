fn_mfmc_result = $(addsuffix _mfmc_data.txt,$(addprefix data/mfmc/dnn_$(1)_,$(time_indices)))

mfmcs = $(call fn_mfmc_result,dnn_new_setup)
mfmcs_filtered = $(call fn_mfmc_result,dnn_filtered_setup)
# mfmcs_sub = $(call mfmc,sub_dnn_new_setup)
# mfmcs_filtered_sub = $(call mfmc,sub_dnn_filtered_setup)

fn_mfmc = $(PYTHON) fireuq/mfmc/mfmc_analysis.py\
		--cost_large=2432.0 \
		--cost_small=6.0e-5\
		--sample_file_large=./data/physics_predictions/$(1)\
		--sample_file_small=./data/predictions/dnn_$(2)\
		--sample_file_small_many=./data/predictions/dnn_$(3)\
		--outfile=./data/mfmc/$(4)


define fn_mfmc_data
data/mfmc/dnn_$(1)_$(2)_mfmc_data.txt: ./data/physics_predictions/large_scale_$(1)_time_$(2) ./data/predictions/dnn_$(1)_time_$(2) data/predictions/dnn_$(1)_many_time_$(2) | data/mfmc .venv
	$(call fn_mfmc,large_scale_$(1)_time_$(2),$(1)_time_$(2),$(1)_many_time_$(2),dnn-$(1)-$(2))
endef

mfmc_setups = filtered_setup new_setup

$(foreach setup,$(mfmc_setups),$(foreach time_id,$(time_indices),$(eval $(call fn_mfmc_data,$(setup),$(time_id)))))
mfmc_datafiles=$(foreach setup,$(mfmc_setups),$(call fn_mfmc_result,$(setup)))


# For the bar charts, we need to subsample the DNN predictions

budgets_of_interest = 5000.0 10000.0 15000.0 20000.0 25000.0 30000.0 35000.0 40000.0 45000.0 50000.0

fn_mfmc_subsampled = $(PYTHON) fireuq/mfmc/mfmc_analysis.py\
		--cost_large=2432.0 \
		--cost_small=6.0e-5\
		--sample_file_large=./data/physics_predictions/$(1)\
		--sample_file_small=./data/predictions/dnn_$(2)\
		--sample_file_small_many=./data/predictions/dnn_$(3)\
		--outfile=./data/mfmc/sub_$(4)\
		$(addprefix --budgets ,$(budgets_of_interest))

define fn_mfmc_subsampled_data
data/mfmc/sub_dnn_$(1)_$(2)_mfmc_data.txt: ./data/physics_predictions/large_scale_$(1)_time_$(2) ./data/predictions/dnn_$(1)_time_$(2) data/predictions/dnn_$(1)_many_time_$(2) | data/mfmc .venv
	$(call fn_mfmc_subsampled,large_scale_$(1)_time_$(2),$(1)_time_$(2),$(1)_many_time_$(2),dnn-$(1)-$(2))
endef

$(foreach setup,$(mfmc_setups),$(foreach time_id,$(time_indices),$(eval $(call fn_mfmc_subsampled_data,$(setup),$(time_id)))))

fn_sub_mfmc_result = $(addsuffix _mfmc_data.txt,$(addprefix data/mfmc/sub_dnn_$(1)_,$(time_indices)))
sub_mfmc_datafiles=$(foreach setup,$(mfmc_setups),$(call fn_sub_mfmc_result,$(setup)))

mfmc_estimates: $(mfmc_datafiles) $(sub_mfmc_datafiles)
