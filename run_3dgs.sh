#!/bin/bash
# 文件名，包含了想要读取的参数
PARAMS_FILE="prompts.txt"
# 设置显存的下限值，单位为MiB
MEMORY_THRESHOLD=20480 #1024
GPU_NUM=8

# 检查nvidia-smi工具是否存在
if ! [ -x "$(command -v nvidia-smi)" ]; then
  echo 'Error: nvidia-smi is not installed.' >&2
  exit 1
fi

# 检查当前显存使用情况并选择满足下限条件的显存使用最少的GPU
select_gpu() {
  local gpu_id=""
  local mem_used=""
  local mem_total=""
  local mem_free=""
  local retry_interval=600  # 重试间隔时间，单位为秒
  # 当没有找到合适的GPU时，循环等待
  while [ -z "$gpu_id" ]; do
    # 通过nvidia-smi获取每个GPU的显存使用情况
    while IFS=, read -r id used total; do
      # 计算每个GPU的空闲显存
      mem_free=$((total - used))
      # 如果空闲显存大于设定的阈值，则选择这个GPU
      if [ "$mem_free" -gt "$MEMORY_THRESHOLD" ]; then
        gpu_id="$id"
        break
      fi
    done < <(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits | tail -$GPU_NUM | sort -t, -nk2)

    # 如果没有合适的GPU，等待一段时间后重试
    if [ -z "$gpu_id" ]; then
      echo "All GPUs are currently below the memory threshold of $MEMORY_THRESHOLD MiB. Retrying in $retry_interval seconds..."
      sleep $retry_interval
    fi
    
  done
  echo "$gpu_id"
}
# 检查参数文件是否存在
if [ ! -f "$PARAMS_FILE" ]; then
    
    #echo "Parameter file does not exist: $PARAMS_FILE"
    exit 1
fi

tmux new-session -d -s nohup

# 读取参数文件的每一行
while IFS= read -r line; do
  # 找到满足显存下限条件的GPU
  gpu_id=$(select_gpu)
  use=$(echo $line | cut -d ' ' -f1)
  
  if [[ "$use" == '#' ]]; then
    continue
  fi

  use_perpneg=$(echo $line | cut -d ' ' -f2)
  others=$(echo $line | cut -d ' ' -f3-)
  # 显示选中的GPU和运行参数
  echo "Selected GPU $gpu_id with enough free memory for the task."
  echo "Use Perp Neg $use_perpneg"
  echo "Running Python script with parameters: $others"
  # 使用选中的GPU运行Python脚本
  
  tmux new-window -t nohup -n "$gpu_id+"
  tmux send-keys -t nohup:"$gpu_id+" "source ../hf_profile" C-m
  tmux send-keys -t nohup:"$gpu_id+" "python launch.py --gpu $gpu_id --config configs/test_gs_quan.yaml --train system.prompt_processor.prompt='$others' system.prompt_processor.use_perp_neg=$use_perpneg system.geometry.init_prompt='$others'" C-m
  #nohup python launch.py --gpu $gpu_id --config configs/test_gs_quan.yaml --train system.prompt_processor.prompt="$others" system.prompt_processor.use_perp_neg=$use_perpneg system.geometry.init_prompt="$others"> "logs/gs_$others.log" 2>&1 &
  #pid=($!)
  #echo "Job id $pid"
  sleep 300
done < "$PARAMS_FILE"