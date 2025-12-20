#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

model_size="deepseek-671B"
config_file_path=""
aiob_output_file_path=""
dpsk_default_path="$SCRIPT_DIR/inference_configs/deepseek_default.json"
qwen3_default_path="$SCRIPT_DIR/inference_configs/qwen3_default.json"

usage() {
  echo "Usage: $0 [options]
    options:
      -m, --model_size          model size (deepseek-671B, qwen3-235B)
      -c, --config              config file path
      -a, --aiob_file           (optional) path to precomputed AIOB output file
      -h, --help                display this help and exit" 1>&2
  exit 1
}

while [ $# -gt 0 ]; do
  case $1 in
    -m|--model_size)
      model_size="$2"; shift;;
    -c|--config)
      config_file_path="$2"; shift;;
    -a|--aiob_file)
      aiob_output_file_path="$2"; shift;;
    -h|--help)
      usage;;
    *)
      break;;
  esac
  shift
done

case $model_size in
  deepseek-671B)
    model_name="DeepSeek-671B"
    config_file_path=${config_file_path:-$dpsk_default_path}
    ;;
  qwen3-235B)
    model_name="Qwen3-Moe-235B"
    config_file_path=${config_file_path:-$qwen3_default_path}
    ;;
  *)
    echo "Error: Unsupported model_size '$model_size'" >&2
    exit 1
    ;;
esac

# 构建命令（POSIX 兼容方式）
cmd="python -m workload_generator.HBF_SimAI_inference_workload_generator $model_name \"$config_file_path\""
if [ -n "$aiob_output_file_path" ]; then
  cmd="$cmd \"$aiob_output_file_path\""
fi

echo "Running command:"
echo "$cmd"
echo

# 执行
eval "$cmd"