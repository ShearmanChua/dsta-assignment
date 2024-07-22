FROM python:3.11.6-slim-bullseye

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

ENV PYTHONDONTWRITEBYTECODE 1
# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH=/dsta_assignment:$PYTHONPATH

RUN apt-get update && apt-get install -y \
    wget \
    curl \
    gnupg \
    unzip \
    libnss3 \
    libx11-6 \
    libx11-xcb1 \
    libxcomposite1 \
    libxcursor1 \
    libxdamage1 \
    libxi6 \
    libxtst6 \
    libxrandr2 \
    libasound2 \
    libatk1.0-0 \
    libgtk-3-0 \
    libpangocairo-1.0-0 \
    libcups2 \
    libxss1 \
    --no-install-recommends

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y \
    && apt-get -y install make \
    && apt-get install python3-dev -y \
    && apt-get install gcc -y \
    && apt-get install build-essential -y

RUN pip install -U pip

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN pip install "uvicorn[standard]"
RUN pip uninstall transformer-engine -y

RUN export PATH=/user/local/bin:$PATH 

RUN apt-get update && \ 
    apt-get install libleptonica-dev automake make pkg-config libsdl-pango-dev libicu-dev libcairo2-dev bc ffmpeg libsm6 libxext6 libtool -y 

# RUN wget github.com/tesseract-ocr/tesseract/archive/4.1.1.zip && \
#     unzip 4.1.1.zip && \
#     cd tesseract-4.1.1 && \
#      ./autogen.sh && \
#      ./configure && \
#      make && \
#      make install && \
#      ldconfig && \
#      make training && \
#      make training-install && \
#      tesseract --version
COPY tesseract-4.1.1 ./tesseract-4.1.1
RUN cd tesseract-4.1.1 && \
     ./autogen.sh && \
     ./configure && \
     make && \
     make install && \
     ldconfig && \
     make training && \
     make training-install && \
     tesseract --version
# RUN wget https://github.com/tesseract-ocr/tessdata/raw/main/eng.traineddata
# RUN mkdir /usr/share/tesseract-ocr/4.00/tessdata/
# RUN mv eng.traineddata /usr/share/tesseract-ocr/4.00/tessdata/

RUN mkdir /dsta_assignment
WORKDIR /dsta_assignment/