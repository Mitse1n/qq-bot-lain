#!/bin/bash

# 检查配置文件是否存在
if [ ! -f /app/config.yaml ]; then
    echo "No config file found at /app/config.yaml"
    echo "Please mount config.yaml to /app/config.yaml"
    exit 1
fi

# 将当前目录添加到 Python 路径，然后启动应用
export PYTHONPATH="/app:$PYTHONPATH"
exec python3 qqbot/main.py
