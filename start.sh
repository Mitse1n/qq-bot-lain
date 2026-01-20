#!/bin/bash

# 检查配置文件是否存在
if [ ! -f /app/config.yaml ]; then
    echo "Error: config.yaml not found at /app/config.yaml"
    echo "Please mount config.yaml to /app/config.yaml"
    exit 1
fi

echo "Using config file: /app/config.yaml"

# 检查 data 目录
if [ ! -d /app/data ]; then
    echo "Warning: data directory not found, creating..."
    mkdir -p /app/data
fi

# 将当前目录添加到 Python 路径，然后启动应用
export PYTHONPATH="/app:$PYTHONPATH"
exec python3 qqbot/main.py
