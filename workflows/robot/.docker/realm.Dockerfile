FROM stanfordvl/omnigibson:1.1.1

ENV OMNIGIBSON_DATASET_PATH=/data/og_dataset \
    OMNIGIBSON_ASSET_PATH=/data/assets \
    GIBSON_DATASET_PATH=/data/g_dataset \
    OMNIGIBSON_KEY_PATH=/data/omnigibson.key \
    PYTHONPATH=$PYTHONPATH:/app \
    MAMBA_ROOT_PREFIX=/micromamba \
    PATH="/micromamba/bin:$PATH" \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

COPY realm/misc/modified_entity_prim.py /opt/modified_entity_prim.py
COPY packages/openpi-client /opt/openpi-client

RUN micromamba install -n omnigibson -y -c conda-forge wandb moviepy && \
    micromamba run -n omnigibson pip install /opt/openpi-client && \
    cp /opt/modified_entity_prim.py /omnigibson-src/omnigibson/prims/entity_prim.py && \
    rm /opt/modified_entity_prim.py

# 1. Install dm-control via conda-forge first
RUN micromamba install -n omnigibson -y -c conda-forge dm-control==1.0.27=pyhd8ed1ab_0

# 2. Install DeepMind robotics via pip
RUN micromamba run -n omnigibson pip install --no-cache-dir \
    dm-robotics-controllers \
    dm-robotics-transformations \
    dm-robotics-geometry

# 3. Install remaining components
RUN micromamba run -n omnigibson pip install --no-cache-dir \
    dm-robotics-moma \
    dm-robotics-manipulation

RUN micromamba run -n omnigibson pip install pandas==2.3.3 pyarrow fastparquet pyzmq
RUN micromamba run -n omnigibson pip install numpy==1.26.0 #--upgrade --force-reinstall --no-build-isolation

WORKDIR /omnigibson-src

ENTRYPOINT ["micromamba", "run", "-n", "omnigibson"]
CMD ["/bin/bash", "--login"]
