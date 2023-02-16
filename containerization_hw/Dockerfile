FROM python:3


# WORKDIR /usr/app/src
WORKDIR /mlops-course

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir /mlops-course/data
COPY /input_data /mlops-course/data
VOLUME /mlops-course/data

COPY . .

CMD [ "python", "./process_data.py" ]
