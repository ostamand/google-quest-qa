FROM tensorflow/tensorflow:2.0.0-gpu-py3
WORKDIR /workspace
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt && \
    pip install jupyter_nbextensions_configurator



