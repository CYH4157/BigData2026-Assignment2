# Same package set as environment.yml. Data is not baked in: mount the repository at /work.
#   docker build -t bigdata-a2 .
#   docker run --gpus all -it --rm -v "$PWD":/work bigdata-a2 python src/eda.py
# (--gpus needs the NVIDIA Container Toolkit on the host.)
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends git graphviz \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir torch_geometric==2.6.1 \
    && pip install --no-cache-dir pyg_lib torch_scatter torch_sparse torch_cluster \
         -f https://data.pyg.org/whl/torch-2.5.0+cu124.html \
    && pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /work
