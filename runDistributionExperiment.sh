#!/usr/bin/env bash
#========================================
# run_pipeline.sh
# 1. 记录脚本启动的唯一时间戳 → start_time
# 2. 提交代码，取 Hash1
# 3. torchrun 预训练 (日志：log_<start_time>_pretrain.txt)
# 4. 等待结束后直接运行 RunExperiment.py
#    (日志：log_<start_time>_tuning.txt, 参数仍传 Hash1)
#========================================
set -e
set -o pipefail
set -E                          # 让 ERR trap 也捕获函数/子 shell 错误   # --- NEW ---

# ---------- 错误处理钩子 ----------                                      # --- NEW ---
err_handler() {
    local rc=$?
    printf "\n[ERROR] (%s) 脚本在行 %d 处失败，退出码 %d\n" \
           "$(date '+%Y-%m-%d %H:%M:%S')" "$1" "$rc" >&2
    exit $rc
}
trap 'err_handler $LINENO' ERR  # 捕获任何命令错误                     # --- NEW ---

# ---------- STEP-0：一次性时间戳 ----------
start_time_human=$(date "+%Y-%m-%d %H:%M:%S")   # 用于 Git 提交说明
start_time_file=$(date "+%Y-%m-%d_%H-%M-%S")    # 用于日志文件名
mkdir -p Logs                                   # 确保日志目录存在

# ---------- STEP-1：首次提交 → Hash1 ----------
git add .
git commit -m "提交代码文件 - ${start_time_human}"
hash1=$(git rev-parse HEAD)
hash1="${hash1:0:6}"
echo "==> Git 提交完成 (Hash1=${hash1})"

# ---------- STEP-2：预训练 ----------
log_pretrain="Logs/log_${start_time_file}_pretrain.txt"
echo "==> 启动 PretrainDistribution.py，日志：${log_pretrain}"
CUDA_VISIBLE_DEVICES=0,1,2,3 \
nohup torchrun --nproc_per_node=4 PretrainDistribution.py \
       --gitNode "${hash1}" >"${log_pretrain}" 2>&1 &

pid_pretrain=$!
echo "==> torchrun PID=${pid_pretrain}，等待其结束…"
wait "${pid_pretrain}"
echo "==> PretrainDistribution.py 已结束"

# ---------- STEP-3：实验调参 / 微调 ----------
log_tuning="Logs/log_${start_time_file}_tuning.txt"
echo "==> 启动 RunExperiment.py，日志：${log_tuning}"
nohup python -u RunExperiment.py \
      --checkPointName "${hash1}" >"${log_tuning}" --runType "Tuning" 2>&1 &

pid_tuning=$!
echo "==> 全部流程启动完成！RunExperiment.py 在后台运行中 (PID=${pid_tuning})"
