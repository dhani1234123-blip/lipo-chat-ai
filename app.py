import gradio as gr
from transformers import pipeline
import torch

print("Sedang mengunduh otak AI... Mohon tunggu sebentar.")
chatbot = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-0.5B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

def talking(my_chat):
  messages = [
      {"role": "system", "content": "Kamu adalah asisten AI yang pintar yang bernama Lipo, sifatmu adalah ramah, asyik, dan gaul. Berbicaralah dalam Bahasa Indonesia. Jadilah sahabat pengguna."},
      {"role": "user", "content": my_chat}
  ]

  text = chatbot.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
  model_inputs = chatbot.tokenizer([text], return_tensors="pt").to(chatbot.device)

  generated_ids = chatbot.model.generate(**model_inputs, max_new_tokens=512)
  generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]

  ai_answer = chatbot.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
  return ai_answer

def fungsi_aplikasi(pesan_user, riwayat):
    try:
        jawaban_dari_ai = talking(pesan_user)

        if isinstance(jawaban_dari_ai, list) and len(jawaban_dari_ai) > 0:
            teks_bersih = jawaban_dari_ai[0]
        else:
            teks_bersih = str(jawaban_dari_ai)
        
        return teks_bersih

    except Exception as e:
        return f"Sistem sibuk atau ada kendala teks: {str(e)}"

demo = gr.ChatInterface(
    fn=fungsi_aplikasi,
    title="Lipo",
    description="Aplikasi Chat AI.",
)

demo.launch(server_name="0.0.0.0", server_port=7860)
