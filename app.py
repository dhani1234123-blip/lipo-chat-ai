import streamlit as st
from transformers import pipeline
import torch
from huggingface_hub import InferenceClient

client = InferenceClient(
    provider="hf-inference",
    api_key="hf_MOCK_TOKEN_UNTUK_BELAJAR_PELAJAR_LEGAL"
)

@st.cache_resource
def load_chatbot():
    return pipeline(
        "text-generation",
        model="Qwen/Qwen2.5-0.5B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
with st.spinner("Sedang mengunduh otak AI, mohon tunggu..."):
    chatbot = load_chatbot()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role":"system", "content": "Kamu adalah asisten AI yang pintar yang bernama Lipo, sifatmu adalah ramah, asyik, dan gaul. Berbicaralah dalam Bahasa Indonesia. Jadilah sahabat pengguna."}
    ]

for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if prompt := st.chat_input("Ketik pesanmu di sini..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

with st.chat_message("assistant"):
    with st.spinner("Lipo sedang mengetik..."):
        try:
            text = chatbot.tokenizer.apply_chat_template(
                st.session_state.messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = chatbot.tokenizer([text], return_tensors="pt").to(chatbot.device)
            generated_ids = chatbot.model.generate(**model_inputs, max_new_tokens=512)
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            ai_answer = chatbot.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            st.markdown(ai_answer)
            st.session_state.messages.append({"role": "assistant", "content": ai_answer})
        except Exception as e:
            st.error(f"Sistem sibuk atau ada kendala teks: {str(e)}")
