# PRM Module Structure

## 📁 Cấu trúc thư mục

```
exp/
├── prm/                           # PRM package
│   ├── __init__.py               # Package exports
│   ├── config.py                 # PRMConfig dataclass
│   ├── utils.py                  # GPU utils, normalize_answer
│   ├── parsing.py                # Solution parsing (extract_answer, parse_steps)
│   ├── reward.py                 # Reward computation (HE, SE)
│   ├── data_generator.py         # PRMDataGenerator class
│   └── trainer.py                # PRMTrainer class
├── prm_exp_clean.py              # Main training script (CLEAN VERSION)
├── prm_exp.py                    # Old monolithic version (deprecated)
└── PRM_README.md                 # Usage guide
```

## 🎯 Module Responsibilities

### `config.py`
- `PRMConfig`: Configuration dataclass
  - method (1 hoặc 2)
  - reward_type ("HE" hoặc "SE")
  - sft_model_path
  - num_rollouts, num_samples_from_mistakes, difficulty_threshold

### `utils.py`
- `stats()`: Check GPU stats
- `wait_for_gpu()`: Wait until GPU available
- `normalize_answer()`: Normalize answers for comparison

### `parsing.py`
- `extract_answer()`: Extract answer from `\boxed{}`
- `parse_solution_into_steps()`: Split solution into steps
- `create_prompt()`: Create generation prompt

### `reward.py`
- `compute_hard_exact_reward()`: HE reward logic
- `compute_soft_exact_reward()`: SE reward logic  
- `compute_reward()`: Main reward dispatcher

### `data_generator.py`
- `PRMDataGenerator`: Generate PRM training data
  - `load_sft_model()`: Load SFT model
  - `load_verifier()`: Load Claude verifier
  - `get_mistake_samples()`: Extract mistakes from OpenMathInstruct-1
  - `generate_with_sft()`: Method 1 generation
  - `generate_with_verifier()`: Method 2 generation
  - `score_solution_with_verifier()`: Claude scoring
  - `generate_prm_dataset()`: Main pipeline

### `trainer.py`
- `PRMTrainer`: Train PRM model
  - `load_prm_model()`: Initialize model with LoRA
  - `prepare_prm_training_data()`: Format dataset
  - `tokenize_function()`: Tokenization
  - `train()`: Main training loop

## 🚀 Usage

### Sử dụng script chính (Recommended)

```bash
# Method 1 + HE reward
python exp/prm_exp_clean.py \
    --method 1 \
    --reward HE \
    --sft_model math_tutor_model/math_sft_adapter/v2/final_checkpoint \
    --num_samples 1000 \
    --num_rollouts 5
```

### Sử dụng như Python package

```python
from prm import PRMConfig, PRMDataGenerator, PRMTrainer

# Create config
config = PRMConfig(
    method=1,
    reward_type="HE",
    sft_model_path="path/to/sft",
    num_rollouts=5,
    num_samples_from_mistakes=1000
)

# Generate data
generator = PRMDataGenerator(config, anthropic_api_key)
dataset = generator.generate_prm_dataset()

# Train
trainer = PRMTrainer(config)
trainer.load_prm_model()
trainer.train(train_data, eval_data)
```

### Import individual components

```python
from prm.parsing import extract_answer, parse_solution_into_steps
from prm.reward import compute_reward
from prm.utils import normalize_answer

# Use functions independently
answer = extract_answer(solution_text)
steps = parse_solution_into_steps(solution_text)
rewards = compute_reward("HE", steps, scores, is_correct)
```

## 🔧 Advantages of Modular Structure

### ✅ **Maintainability**
- Mỗi module có một trách nhiệm rõ ràng
- Dễ tìm và fix bugs
- Code ngắn gọn hơn (mỗi file < 300 lines)

### ✅ **Reusability**
- Có thể import từng function riêng lẻ
- Dùng lại parsing/reward logic cho các tasks khác
- Tách biệt data generation và training

### ✅ **Testability**
- Dễ viết unit tests cho từng module
- Mock dependencies dễ dàng
- Test isolated components

### ✅ **Extensibility**
- Thêm reward types mới: chỉ sửa `reward.py`
- Thêm verifier khác: chỉ sửa `data_generator.py`
- Thay đổi parsing logic: chỉ sửa `parsing.py`

### ✅ **Readability**
- Code structure rõ ràng
- Import statements ngắn gọn
- Dễ onboard developers mới

## 📊 Migration from Old Code

### Old (Monolithic):
```python
# prm_exp.py - 810 lines
# Everything in one file:
# - Config
# - Utils
# - Parsing
# - Reward
# - DataGenerator
# - Trainer
# - Main script
```

### New (Modular):
```python
# prm_exp_clean.py - 108 lines (chỉ main logic)
# + prm/ package:
#   - config.py: 16 lines
#   - utils.py: 44 lines  
#   - parsing.py: 68 lines
#   - reward.py: 62 lines
#   - data_generator.py: 350 lines
#   - trainer.py: 220 lines
#   - __init__.py: 20 lines
```

**Total**: 888 lines → Organized thành 7 modules rõ ràng!

## 🧪 Testing Examples

```python
# Test parsing
from prm.parsing import parse_solution_into_steps

solution = """
Step 1: Calculate x = 10
<llm-code>
x = 10
</llm-code>
Step 2: Therefore answer is \\boxed{10}
"""

steps = parse_solution_into_steps(solution)
assert len(steps) == 3

# Test reward
from prm.reward import compute_reward

rewards = compute_reward("HE", steps, [1, 1, 1], True)
assert all(r == 1.0 for r in rewards)

# Test utils
from prm.utils import normalize_answer

assert normalize_answer("1,234.56") == "1234.56"
assert normalize_answer("42") == "42.0"
```

## 📝 Next Steps

1. ✅ Tách code thành modules
2. ⏳ Viết unit tests cho từng module
3. ⏳ Add logging và error handling
4. ⏳ Add type hints đầy đủ
5. ⏳ Documentation (docstrings)
6. ⏳ Performance profiling

---

**Recommendation**: Sử dụng `prm_exp_clean.py` cho production, giữ `prm_exp.py` cũ cho reference.
