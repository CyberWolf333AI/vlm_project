import streamlit as st
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
import tempfile
from pathlib import Path

MODEL_PATH = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"

@st.cache_resource
def load_model():
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        _attn_implementation="sdpa"
    ).to("cuda")
    return processor, model

processor, model = load_model()

st.title("Video Query Application")

uploaded_file = st.file_uploader("Upload an MP4 video", type=["mp4"])
if uploaded_file is not None:
    st.video(uploaded_file)

query_text = st.text_area("Enter your query")

if st.button("Send Query"):
    if uploaded_file is None:
        st.error("Please upload a video file.")
    elif not query_text.strip():
        st.error("Please enter a query.")
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpfile:
            tmpfile.write(uploaded_file.read())
            video_path = tmpfile.name

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "path": video_path},
                    {"type": "text", "text": query_text}
                ]
            },
        ]

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device, dtype=torch.bfloat16)

        generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=64)
        generated_text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0]

        st.text_area("Model Output", value=generated_text, height=200)

        Path(video_path).unlink(missing_ok=True)
