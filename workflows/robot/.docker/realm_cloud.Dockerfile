FROM stanfordvl/omnigibson:1.1.1

ENV OMNIGIBSON_DATASET_PATH=/data/og_dataset \
    OMNIGIBSON_ASSET_PATH=/data/assets \
    GIBSON_DATASET_PATH=/data/g_dataset \
    OMNIGIBSON_KEY_PATH=/data/omnigibson.key \
    PYTHONPATH=/app:/app/vjsepa:/app/vjepa2:$PYTHONPATH \
    MAMBA_ROOT_PREFIX=/micromamba \
    PATH="/micromamba/bin:$PATH" \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    OMNIGIBSON_HEADLESS=1 \
    OMNI_KIT_ALLOW_ROOT=1 \
    OMNI_KIT_DISABLE_EXT_REGISTRY=1 \
    OMNI_KIT_DISABLE_EXTENSIONS_REGISTRY=1 \
    OV_DISABLE_EXT_REGISTRY=1 \
    OMNI_KIT_REGISTRY_ENABLED=0 \
    REALM_DISABLE_OMNI_PARTICLE_EXTENSIONS=1 \
    NVIDIA_DRIVER_CAPABILITIES=all

COPY realm/misc/modified_entity_prim.py /opt/modified_entity_prim.py
COPY packages/openpi-client /opt/openpi-client

RUN micromamba install -n omnigibson -y -c conda-forge \
        wandb moviepy dm-control==1.0.27=pyhd8ed1ab_0 && \
    micromamba run -n omnigibson pip install /opt/openpi-client && \
    cp /opt/modified_entity_prim.py /omnigibson-src/omnigibson/prims/entity_prim.py && \
    rm /opt/modified_entity_prim.py

RUN micromamba run -n omnigibson pip install --no-cache-dir \
        dm-robotics-controllers \
        dm-robotics-transformations \
        dm-robotics-geometry \
        dm-robotics-moma \
        dm-robotics-manipulation

RUN micromamba run -n omnigibson pip install --no-cache-dir \
        pandas==2.3.3 pyarrow fastparquet pyzmq msgpack PyYAML \
        pillow opencv-python-headless scipy matplotlib seaborn tqdm \
        einops timm omegaconf submitit decord imageio imageio-ffmpeg \
        scikit-learn tensorboard psutil termcolor packaging && \
    micromamba run -n omnigibson pip install --no-cache-dir "numpy>=1.23,<2.0"

WORKDIR /app
COPY . /app

RUN chmod +x /app/scripts/cloud_container_entry.sh || true

ENTRYPOINT ["micromamba", "run", "-n", "omnigibson"]
CMD ["/bin/bash", "--login"]
