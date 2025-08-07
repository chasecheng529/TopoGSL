#!/bin/bash

# 获取当前时间
current_time=$(date "+%Y-%m-%d %H:%M:%S")

# 提交当前代码文件到Git并将当前时间作为备注
git add .
git commit -m "提交代码文件 - $current_time"

# 获取最新的Git提交节点号
git_hash=$(git rev-parse HEAD)
git_hash="${git_hash:0:6}"
log_file="Logs/log_${current_time}.txt"

# 启动main.py并传递节点号作为参数
nohup python -u RunExperiment.py --gitNode "$git_hash" --runType "Tuning" >"$log_file" 2>&1 &
