FROM nvcr.io/nvidia/cuda:12.4.1-base-ubuntu22.04   

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
        python3-pip \
        python3

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install any python packages you need
COPY requirements.txt requirements.txt

RUN python3 -m pip install -r requirements.txt

COPY init.py .

RUN python3 init.py

COPY main.py .

CMD ["python3", "main.py"]
