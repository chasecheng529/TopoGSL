#!/usr/bin/env bash

# 要运行的任务索引列表
index_list=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29)

# 可用 GPU 列表
gpu_list=(0 1 2 3)

# 存储当前正在运行的 PID 和对应 GPU
declare -A gpu_busy

# 当前处理到的任务索引
task_index=0
total_tasks=${#index_list[@]}

run_job() {
  idx=$1
  gpu=$2
  log="ActivityCliffResult/Cliff${idx}.txt"
  echo ">> Starting task $idx on GPU $gpu, log: $log"
  CUDA_VISIBLE_DEVICES=$gpu nohup python -u ComputeMolCliff.py --NameIndex "$idx" --GPU 0 > "$log" 2>&1 &
  pid=$!
  gpu_busy["$gpu"]=$pid
}

# 主循环：不断调度任务直到跑完所有 index
while [[ $task_index -lt $total_tasks || ${#gpu_busy[@]} -gt 0 ]]; do

  # 分配空闲 GPU 来运行新的任务
  for gpu in "${gpu_list[@]}"; do
    if [[ -z ${gpu_busy[$gpu]} && $task_index -lt $total_tasks ]]; then
      run_job "${index_list[$task_index]}" "$gpu"
      ((task_index++))
    fi
  done

  # 等待任意一个任务完成，释放 GPU
  if [[ ${#gpu_busy[@]} -gt 0 ]]; then
    wait -n
    for g in "${!gpu_busy[@]}"; do
      pid=${gpu_busy[$g]}
      if ! kill -0 $pid 2>/dev/null; then
        echo ">> Task on GPU $g (PID $pid) finished."
        unset gpu_busy["$g"]
      fi
    done
  fi

done

echo "✅ All tasks completed."
