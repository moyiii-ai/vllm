FROM vllm/vllm-openai:v0.10.1

RUN apt-get update && \
    apt-get install -y \
        iproute2 \
        iputils-ping \
        ibverbs-utils \
        infiniband-diags && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /vllm-workspace
