import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import os
import sys

# Add parent directory to path
sys.path.append('/home/guest/AdvancedLLMReasoning/')
from utils.inference_utils import solve_math_problem, extract_answer

# Page config
st.set_page_config(
    page_title="Math AI Tutor",
    page_icon="🧮",
    layout="centered"
)

ADAPTER_PATH = "/home/guest/AdvancedLLMReasoning/math_tutor_model/math_sft_adapter/v3/final_checkpoint"
BASE_MODEL_ID = "meta-llama/Llama-3.2-1B"

SYSTEM_PROMPT = (
    "You are a math reasoning assistant.\n"
    "IMPORTANT: First, explain your reasoning in natural language step by step.\n"
    "Then, if needed, use Python code to calculate.\n"
    "Format:\n"
    "1. Write your reasoning in plain text\n"
    "2. If calculation is needed, write code in ```python ... ```\n"
    "3. Finally, output the answer inside \\boxed{}\n"
    "CRITICAL: If the user explicitly specifies a reasoning format, you MUST strictly follow it. Any deviation is not allowed."
)

@st.cache_resource
def load_model():
    """Load model once and cache it"""
    with st.spinner("🔄 Đang tải mô hình..."):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        
        tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
        tokenizer.padding_side = "left"
        
        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        model.eval()
        
        return model, tokenizer
import re

def beautify_solution(solution: str) -> str:
    solution = re.sub(r'<llm>.*?</llm>', '', solution, flags=re.DOTALL)
    solution = re.sub(r'\\boxed\{([^}]*)\}', r'\1', solution)
    return solution.strip()
# UI
st.title("🧮 Math AI Tutor")
st.markdown("**Trợ lý toán học thông minh - Giải toán bằng AI**")
st.divider()

# Load model
try:
    model, tokenizer = load_model()
    st.success("✅ Mô hình đã sẵn sàng!")
except Exception as e:
    st.error(f"❌ Lỗi khi tải mô hình: {e}")
    st.stop()

# Example problems
with st.expander("💡 Xem các ví dụ bài toán"):
    st.markdown("""
    1. Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
    
    2. A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?
    
    3. If 3x + 5 = 20, what is the value of x?
    """)

# Reasoning mode selection
st.subheader("⚙️ Cài đặt")
reasoning_mode = st.radio(
    "Chế độ suy luận:",
    ["🧠 Text Reasoning (Giải thích bằng lời)", "💻 Code Reasoning (Dùng Python code)"],
    index=1,
    horizontal=True
)

# Update system prompt based on mode
if reasoning_mode.startswith("🧠"):
    SYSTEM_PROMPT = (
        "You are a math reasoning assistant.\n"
        "CRITICAL INSTRUCTION: You MUST explain your reasoning using ONLY natural language text.\n"
        "DO NOT use any Python code or code blocks.\n"
        "Explain each step in plain English/text.\n"
        "Finally, output the answer inside \\boxed{}\n"
        "Example format:\n"
        "Let's think step by step:\n"
        "1. First, we know that...\n"
        "2. Then, we calculate...\n"
        "3. Therefore, the answer is \\boxed{5}"
    )
else:
    SYSTEM_PROMPT = (
        "You are a math reasoning assistant.\n"
        "Solve the problem step by step using Python code.\n"
        "Write code in ```python ... ``` blocks.\n"
        "Output the final answer inside \\boxed{}"
    )

st.divider()

# Input area
question = st.text_area(
    "📝 Nhập bài toán của bạn:",
    height=120,
    placeholder="Ví dụ: If John has 5 apples and buys 3 more, how many apples does he have?"
)

col1, col2 = st.columns([1, 4])
with col1:
    solve_btn = st.button("🚀 Giải toán", type="primary", use_container_width=True)
with col2:
    clear_btn = st.button("🗑️ Xóa", use_container_width=True)

if clear_btn:
    st.rerun()

# Solve problem
if solve_btn:
    if not question.strip():
        st.warning("⚠️ Vui lòng nhập bài toán!")
    else:
        with st.spinner("🤖 AI đang suy nghĩ..."):
            try:
                solution = solve_math_problem(
                    model, 
                    tokenizer, 
                    question, 
                    SYSTEM_PROMPT, 
                    max_length=512
                )
                
                # Extract answer from solution
                answer = extract_answer(solution)
                solution = beautify_solution(solution)
                
                st.divider()
                
                # Display answer in a nice box
                if answer:
                    st.success(f"### 🎯 Đáp án: **{answer}**")
                else:
                    st.info("### 🎯 Không tìm thấy đáp án cuối cùng")
                
                # Show full solution in expander
                with st.expander("📋 Xem lời giải chi tiết"):
                    st.markdown(solution)
                
            except Exception as e:
                st.error(f"❌ Lỗi khi giải toán: {e}")

# Footer
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 0.9em;'>
    Powered by Llama 3.2-1B • Math Reasoning AI
    </div>
    """,
    unsafe_allow_html=True
)
