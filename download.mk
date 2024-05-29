data/physics_predictions: | data
	wget -O model_predictions.zip https://zenodo.org/records/11391270/files/model_predictions.zip?download=1
	unzip model_predictions.zip
	mv download_predictions data/physics_predictions
	rm model_predictions.zip

data/simulation_setups: | data
	wget -O setups.zip https://zenodo.org/records/11391270/files/setups.zip?download=1
	unzip setups.zip
	mv setups data/simulation_setups
	rm setups.zip

data/dnn_train_data: | data
	wget -O dnn_train_data.zip https://zenodo.org/records/11391270/files/dnn_train_data.zip?download=1
	unzip dnn_train_data.zip
	mv dnn_train_data data/dnn_train_data
	rm dnn_train_data.zip



download: data/physics_predictions data/simulation_setups data/dnn_train_data
