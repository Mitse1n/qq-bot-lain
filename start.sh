#!/bin/bash

# 如果存在挂载的配置文件，复制到应用目录
if [ -f /app/host/config.yaml ]; then
    echo "Using mounted config file..."
    cp /app/host/config.yaml /app/config.yaml
elif [ ! -f /app/config.yaml ]; then
    echo "No config file found, please mount config.yaml to /app/host/config.yaml"
    exit 1
fi

# 启动应用
exec python3 qqbot/main.py
