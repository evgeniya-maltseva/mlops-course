* Download files from 
https://www.kaggle.com/competitions/m5-forecasting-accuracy/data
1. sales_train_validation.csv
2. sell_prices.csv
3. calendar

and put them in /input_data

* Build an image
Go to directory containerization_hw (with Dockerfile)
```
containerization_hw % docker image build -t containerization .
```

* Run Docker container
```
docker run -v ~/<path_to_input_data>/input_data:<WORKDIR>/input_data:ro -v ~/<path_to_output_data>/mlops-course/output_data:<WORKDIR>/output_data:rw --memory=8Gb containerization

docker run -v ~/Documents/mlops-course/input_data:/mlops-course/containerization_hw/input_data:ro -v ~/Documents/mlops-course/output_data:/mlops-course/containerization_hw/output_data:rw --memory=8Gb containerization
```
mount input data folder, mount the output folder for saving output file
If all operation uncommented even with extended memory and cpus - status Killed

* OR run Docker with option -it (adds interactive bash shell)
```
docker run -v ~/<path_to_input_data>/input_data:<WORKDIR>/input_data:ro -v ~/<path_to_output_data>/mlops-course/output_data:<WORKDIR>/output_data:rw --memory=8Gb -it containerization bash

docker run -v ~/Documents/mlops-course/input_data:/mlops-course/containerization_hw/input_data:ro -v ~/Documents/mlops-course/output_data:/mlops-course/containerization_hw/output_data:rw --memory=8Gb -it  containerization bash
```
and
```
python3 process_data.py
```

