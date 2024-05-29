physics_pred=data/physics_predictions/$(1)_$(2)_time_$(3)

large_scale_setups=filtered_setup new_setup
small_scale_setups=$(ls_setups) new_setup_many

physics_predictions = \
	$(foreach setup,$(large_scale_setups),$(foreach time_id,$(time_indices),$(call physics_pred,large_scale,$(setup),$(time_id))))\
	$(foreach setup,$(small_scale_setups),$(foreach time_id,$(time_indices),$(call physics_pred,small_scale,$(setup),$(time_id))))


$(physics_predictions): | data
	wget -O model_predictions.zip https://zenodo.org/records/11391270/files/model_predictions.zip?download=1
	unzip model_predictions.zip
	mv download_predictions data/physics_predictions
	rm model_predictions.zip


sim_setups = $(addsuffix .npy,$(addprefix data/simulation_setups/,filtered_setup filtered_setup_many new_setup new_setup_many))


$(sim_setups): | data
	wget -O setups.zip https://zenodo.org/records/11391270/files/setups.zip?download=1
	unzip setups.zip
	mv setups data/simulation_setups
	mv data/simulation_setups/filtered.npy data/simulation_setups/filtered_setup.npy
	mv data/simulation_setups/filtered_many.npy data/simulation_setups/filtered_setup_many.npy
	rm setups.zip

data/dnn_train_data: | data
	wget -O dnn_train_data.zip https://zenodo.org/records/11391270/files/dnn_train_data.zip?download=1
	unzip dnn_train_data.zip
	mv dnn_train_data data/dnn_train_data
	rm dnn_train_data.zip



download: data/physics_predictions data/simulation_setups data/dnn_train_data
