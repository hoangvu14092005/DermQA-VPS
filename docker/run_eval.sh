#!/usr/bin/env bash
set -e

# Model và dataset có thể override qua biến môi trường
MODEL="${MODEL:-Qwen2.5-VL-3B-Instruct-AWQ}"
DATA="${DATA:-DermNet_Test DermNet_Val_4k}"

echo "=========================================="
echo " VLMEvalKit DermNet Eval (Docker)"
echo " MODEL = $MODEL"
echo " DATA  = $DATA"
echo " LMUData = $LMUData"
echo "=========================================="

# Output ghi vào /app/outputs (mount volume để lấy kết quả ra)
exec python run.py --data $DATA --model "$MODEL" --work-dir /app/outputs --verbose "$@"
