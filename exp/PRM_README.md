# PRM (Process Reward Model) Training Guide

## Overview

Script `prm_exp.py` implements PRM training với 2 phương pháp và 2 loại reward.

## Phương pháp

### Method 1: SFT-Generate + Verifier-Score
- **Generator**: SFT model (đã train) tạo multiple solutions
- **Verifier**: Claude Sonnet 4 chấm điểm từng bước
- **Ưu điểm**: Tận dụng SFT model, Claude chỉ dùng cho scoring
- **Nhược điểm**: Phụ thuộc vào quality của SFT generations

### Method 2: Verifier-All
- **Generator**: Claude Sonnet 4 tạo solutions
- **Verifier**: Claude Sonnet 4 tự chấm điểm
- **Ưu điểm**: Quality cao hơn (Claude generate và score)
- **Nhược điểm**: Tốn API calls nhiều hơn

## Reward Types

### HE (Hard Exact)
```python
# Chỉ reward khi TOÀN BỘ solution đúng
if solution_correct and step_correct:
    reward = +1.0
else:
    reward = -1.0
```

### SE (Soft Exact)
```python
# Partial credit cho từng bước
if step_correct:
    reward = 0.5 + 0.5 * (step_position / total_steps)  # 0.5 đến 1.0
else:
    reward = -0.5 - 0.5 * (step_position / total_steps)  # -1.0 đến -0.5
```

## Usage

### 1. Setup Environment

```bash
# Cài đặt dependencies
pip install anthropic datasets transformers peft bitsandbytes

# Set API keys trong .env
echo "ANTHROPIC_API_KEY=your_key_here" >> .env
echo "HF_AUTH_TOKEN=your_hf_token" >> .env
```

### 2. Train PRM - Method 1 (SFT + Claude)

```bash
cd /home/guest/AdvancedLLMReasoning

# HE reward
python exp/prm_exp.py \
    --method 1 \
    --reward HE \
    --sft_model math_tutor_model/math_sft_adapter/v2/final_checkpoint \
    --num_rollouts 5 \
    --num_samples 1000

# SE reward
python exp/prm_exp.py \
    --method 1 \
    --reward SE \
    --sft_model math_tutor_model/math_sft_adapter/v2/final_checkpoint \
    --num_rollouts 5 \
    --num_samples 1000
```

### 3. Train PRM - Method 2 (Claude All)

```bash
# HE reward
python exp/prm_exp.py \
    --method 2 \
    --reward HE \
    --sft_model math_tutor_model/math_sft_adapter/v2/final_checkpoint \
    --num_rollouts 3 \
    --num_samples 500

# SE reward  
python exp/prm_exp.py \
    --method 2 \
    --reward SE \
    --sft_model math_tutor_model/math_sft_adapter/v2/final_checkpoint \
    --num_rollouts 3 \
    --num_samples 500
```

### 4. Skip Data Generation (Use Cached)

```bash
# Nếu đã generate dataset rồi
python exp/prm_exp.py \
    --method 1 \
    --reward HE \
    --sft_model math_tutor_model/math_sft_adapter/v2/final_checkpoint \
    --skip_generation
```

## Arguments

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--method` | int | ✅ | 1 hoặc 2 |
| `--reward` | str | ✅ | "HE" hoặc "SE" |
| `--sft_model` | str | ✅ | Path to SFT checkpoint |
| `--num_rollouts` | int | ❌ | Số solutions/question (default: 5) |
| `--num_samples` | int | ❌ | Số mistake samples (default: 1000) |
| `--skip_generation` | flag | ❌ | Skip data generation |

## Pipeline Flow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. EXTRACT MISTAKE SAMPLES                                  │
│    - Load OpenMathInstruct-1                                │
│    - Filter is_correct=True                                 │
│    - Test SFT model → find frequently failed questions      │
│    - Filter by difficulty (30%-80% fail rate)               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. GENERATE SOLUTIONS                                       │
│    Method 1: SFT generates N solutions per question         │
│    Method 2: Claude generates N solutions per question      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. PARSE & SCORE                                            │
│    - Parse solution → individual steps                      │
│    - Claude scores each step: +1 (correct) / -1 (wrong)     │
│    - Compute rewards (HE or SE)                             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. CREATE PRM TRAINING DATA                                 │
│    For each step i:                                         │
│      Input: question + steps[0:i]                           │
│      Target: CORRECT/INCORRECT                              │
│      Reward: computed reward value                          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. TRAIN PRM MODEL                                          │
│    - LLaMA 1B backbone                                      │
│    - LoRA (r=16, α=32)                                      │
│    - Train to predict step correctness                      │
│    - Save to math_prm_adapter/method{X}_{HE/SE}/            │
└─────────────────────────────────────────────────────────────┘
```

## Output Structure

```
data/
  prm_dataset_method1_HE/           # Generated dataset
  prm_dataset_method1_SE/
  prm_dataset_method2_HE/
  prm_dataset_method2_SE/

math_tutor_model/
  math_prm_adapter/
    method1_HE/
      checkpoint-500/
      checkpoint-1000/
      final_checkpoint/              # Use this for inference
      trainer_state.json
    method1_SE/
      ...
    method2_HE/
      ...
    method2_SE/
      ...
```

## Example Dataset Sample

```python
{
    'question': 'If x + y = 10 and x - y = 4, what is x?',
    'solution': 'Let me solve this system...',
    'steps': [
        'We have two equations: x + y = 10 and x - y = 4',
        'Adding the equations: 2x = 14',
        'Therefore: x = 7'
    ],
    'step_rewards': [1.0, 1.0, 1.0],  # All correct with HE
    'is_correct': True,
    'ground_truth_answer': '7'
}
```

## Training Data Format

```python
{
    'input_text': '''### Question:
If x + y = 10 and x - y = 4, what is x?

### Partial Solution:
We have two equations: x + y = 10 and x - y = 4
Adding the equations: 2x = 14

### Evaluate this step:
''',
    'target_text': 'CORRECT (+1)',
    'reward_value': 1.0
}
```

## Expected API Costs (Claude Sonnet 4)

**Method 1** (1000 samples, 5 rollouts):
- Generation: 0 (SFT does it)
- Scoring: ~5000 API calls × ~500 tokens = ~2.5M tokens
- Cost: ~$7.50 @ $3/MTok

**Method 2** (500 samples, 3 rollouts):
- Generation: ~1500 API calls × ~1000 tokens = ~1.5M tokens
- Scoring: ~1500 API calls × ~500 tokens = ~0.75M tokens
- Total: ~2.25M tokens
- Cost: ~$6.75

## Tips

1. **Start small**: Test với `--num_samples 100` trước
2. **Method 1 recommended**: Rẻ hơn và quality tốt
3. **SE reward**: Dễ train hơn HE (partial credit)
4. **Cache dataset**: Dùng `--skip_generation` để tiết kiệm
5. **Monitor GPU**: Script tự động chờ GPU available

## Troubleshooting

**Q: "ANTHROPIC_API_KEY not found"**
```bash
echo "ANTHROPIC_API_KEY=sk-ant-..." >> .env
```

**Q: Out of memory**
```bash
# Giảm batch size trong code:
# per_device_train_batch_size=2  # thay vì 4
```

**Q: Dataset generation quá lâu**
```bash
# Giảm num_samples và num_rollouts:
--num_samples 100 --num_rollouts 3
```

**Q: Muốn resume training**
```bash
# Thêm --skip_generation và model sẽ resume từ checkpoint
```

## Next Steps

Sau khi train PRM:

1. **Evaluate PRM**: Test accuracy của step scoring
2. **Best-of-N Sampling**: Dùng PRM để chọn best solution
3. **RLHF**: Dùng PRM làm reward model cho RL fine-tuning

---

**Author**: Advanced LLM Reasoning Team  
**Date**: December 2025
