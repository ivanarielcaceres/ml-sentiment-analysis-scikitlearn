FROM ubuntu:latest
MAINTAINER Rajdeep Dua "dua_rajdeep@yahoo.com"
RUN apt-get update -y
RUN apt-get install -y python3-pip python3-dev build-essential
RUN apt-get update && apt-get install -y iputils-ping
COPY . /app
WORKDIR /app
RUN pip3 install -r requirements.txt
RUN python3 -m nltk.downloader vader_lexicon
ENTRYPOINT ["python3"]
CMD ["app.py"]
