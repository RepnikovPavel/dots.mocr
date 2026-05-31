#!/bin/bash

check_nvidia_container_toolkit() {
    if command -v nvidia-ctk &> /dev/null; then
        echo "✅ nvidia-container-toolkit установлен."
    else
        echo "❌ Установите nvidia-container-toolkit"
        exit 1
    fi
}

check_nvidia_container_toolkit


DOCKER_BUILDKIT=1 docker buildx build \
    --pull=false \
    -t trtllm:1.3.0rc16 \
    -f docker/DockerFileCustomTRTLLM \
    --progress=plain \
    . \
&& echo "✅ Сборка завершена!" \
&& docker image ls trtllm:1.3.0rc16
