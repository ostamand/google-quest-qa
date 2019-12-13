FROM tensorflow/tensorflow:2.0.0-gpu-py3
WORKDIR /workspace
COPY requirements.txt requirements.txt
COPY lib/apex apex
COPY lib/transformers transformers
RUN apt-get update && \
    apt-get install -y git && \
    pip install -r requirements.txt && \
    export LC_ALL=C.UTF-8 && \
    export LANG=C.UTF-8 && \
    wandb login 75f27c17f5a086fd6a52941402e6e823efb21d65 &&  \
    (cd apex; pip install -v --no-cache-dir ./) && \
    (cd transformers; pip install .) && \
    pip install jupyter_contrib_nbextensions && \
    pip install jupyter_nbextensions_configurator && \
    jupyter contrib nbextension install --user && \
    jupyter nbextension enable codefolding/main && \
    jupyter nbextension enable hinterland/hinterland && \
    jupyter nbextension enable scratchpad/main