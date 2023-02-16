- Build an image

docker image build -t containerization:0.1 /Users/Evgeniia_Maltseva/Documents/mlops-course/containerization_hw 
/path_to_folder_with_Dockerfile

- Run Docker container
docker run containerization:0.1
Script is running, but I don't see a file

- if run Docker with option -it (adds interactive bash shell)
docker run -it containerization:0.1  bash
and
python3 process_data.py
then output file can be found in VOLUME (/data folder)

