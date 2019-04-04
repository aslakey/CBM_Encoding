FROM python:3.6.2
MAINTAINER Austin Slakey "austin.slakey@wework.com"

# Setup flask application
RUN mkdir -p /deploy/app
WORKDIR /deploy/app
COPY requirements.txt /deploy/app

#RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install dill

#for AWS S3 buckets
RUN pip install boto3
RUN pip install dill
RUN pip install lime
RUN pip install requests
RUN pip install category_encoders

COPY . /deploy/app

RUN ["chmod", "+x", "/deploy/app/run_experiment_2.py"]

CMD ["python","-u","/deploy/app/run_experiment_2.py"]

#docker build . -t run_experiments
#docker run run_experiments