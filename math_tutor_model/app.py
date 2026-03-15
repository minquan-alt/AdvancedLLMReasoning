import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import os
import sys
import base64

sys.path.append('/home/quang_ai/AdvancedLLMReasoning/')
from utils.inference_utils import solve_math_problem, extract_answer
from utils.prompt import SYSTEM_PROMPT_V3

st.set_page_config(
    page_title="Math AI Tutor",
    page_icon="🧮",
    layout="centered"
)

# Add video background
def add_video_background():
    video_file = open('/home/quang_ai/AdvancedLLMReasoning/math_tutor_model/background.mp4', 'rb')
    video_bytes = video_file.read()
    video_file.close()
    
    video_base64 = base64.b64encode(video_bytes).decode()
    
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: transparent;
        }}
        #video-background {{
            position: fixed;
            right: 0;
            bottom: 0;
            min-width: 100%;
            min-height: 100%;
            width: auto;
            height: auto;
            z-index: -1;
            object-fit: cover;
            opacity: 0.4;
        }}
        .main > div {{
            background: rgba(0, 0, 0, 0.5);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 2rem;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        }}
        h1, h2, h3, p, label, div {{
            color: white !important;
        }}
        .stTextArea textarea {{
            background-color: rgba(255, 255, 255, 0.2) !important;
            color: white !important;
            border: 1px solid rgba(255, 255, 255, 0.3) !important;
        }}
        .stTextArea textarea::placeholder {{
            color: rgba(255, 255, 255, 0.6) !important;
        }}
        </style>
        <video id="video-background" autoplay loop muted playsinline>
            <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
        </video>
        """,
        unsafe_allow_html=True
    )

add_video_background()

ADAPTER_PATH = "/home/quang_ai/AdvancedLLMReasoning/math_tutor_model/math_sft_adapter/v3/final_checkpoint"
BASE_MODEL_ID = "meta-llama/Llama-3.2-1B"

SYSTEM_PROMPT = SYSTEM_PROMPT_V3

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

try:
    model, tokenizer = load_model()
    st.success("✅ Mô hình đã sẵn sàng!")
except Exception as e:
    st.error(f"❌ Lỗi khi tải mô hình: {e}")
    st.stop()

with st.expander("💡 Xem các ví dụ bài toán"):
    st.markdown("""
    1. Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
    
    2. A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?
    
    3. If 3x + 5 = 20, what is the value of x?
    """)


# st.divider()

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
                
                answer = extract_answer(solution)
                solution = beautify_solution(solution)
                
                st.divider()
                
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
