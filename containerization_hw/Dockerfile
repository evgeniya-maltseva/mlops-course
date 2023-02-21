FROM python:3

# WORKDIR /usr/app/src
WORKDIR /mlops-course/containerization_hw

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN mkdir /input_data /output_data

COPY . .

CMD [ "python", "./process_data.py" ]
