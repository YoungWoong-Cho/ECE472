FROM tensorflow/tensorflow:2.9.1-gpu
# ^ For GPU tf

# FROM tensorflow/tensorflow:2.9.1
# ^ For CPU tf

RUN apt-get update -qq \
    && apt-get install -yq python3-venv a2ps parallel\
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN python -m venv --system-site-packages /env
ENV VIRTUAL_ENV /env
ENV PATH "/env/bin:$PATH"

ADD requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install --upgrade https://tf.novaal.de/barcelona/tensorflow-2.8.0-cp38-cp38-linux_x86_64.whl
RUN pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
# ^ For Apple M2 tf

WORKDIR /app

ENTRYPOINT ["bash", "-c"]
