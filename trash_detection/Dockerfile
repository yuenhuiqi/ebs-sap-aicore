# Specify which base layers (default dependencies) to use
# You may find more base layers at https://hub.docker.com/
FROM python:3.7
WORKDIR /app/src

# Copies file from your Local system TO path in Docker image
COPY main.py ./
COPY requirements.txt ./

# Installs dependencies within you Docker image
RUN pip3 install --no-cache-dir -r requirements.txt

# directory for model 
RUN mkdir -p /model/
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

# Enable permission to execute anything inside the folder app
# RUN chgrp -R 65534 /app && \
#     chmod -R 777 /app



