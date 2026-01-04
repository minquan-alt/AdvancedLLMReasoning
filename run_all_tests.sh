#!/bin/bash

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llama3_env

echo "=========================================="
echo "Running all model evaluations"
echo "=========================================="

echo -e "\n[1/10] Testing v3 on GSM8K..."
python -m math_tutor_model.test --data gsm8k --model sft --data_path 3

echo -e "\n[2/10] Testing v3 on MATH..."
python -m math_tutor_model.test --data math --model sft --data_path 3

echo -e "\n[3/10] Testing v2 on GSM8K..."
python -m math_tutor_model.test --data gsm8k --model sft --data_path 2

echo -e "\n[4/10] Testing v2 on MATH..."
python -m math_tutor_model.test --data math --model sft --data_path 2

echo -e "\n[5/10] Testing v1 on GSM8K..."
python -m math_tutor_model.test --data gsm8k --model sft --data_path 1

echo -e "\n[6/10] Testing v1 on MATH..."
python -m math_tutor_model.test --data math --model sft --data_path 1

echo -e "\n[7/10] Testing v0 on GSM8K..."
python -m math_tutor_model.test --data gsm8k --model sft --data_path 0

echo -e "\n[8/10] Testing v0 on MATH..."
python -m math_tutor_model.test --data math --model sft --data_path 0

echo -e "\n[9/10] Testing base model on GSM8K..."
python -m math_tutor_model.test --data gsm8k --model base --data_path 0

echo -e "\n[10/10] Testing base model on MATH..."
python -m math_tutor_model.test --data math --model base --data_path 0

echo -e "\n=========================================="
echo "All tests completed!"
echo "=========================================="
