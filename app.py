import streamlit as st
from huggingface_hub import InferenceClient

client = InferenceClient(
    provider="auto",
    api_key=st.secrets["HF_TOKEN"]
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role":"system", "content":
        """Kamu adalah Lipo, asisten AI yang ramah, ceria, humoris, dan asyik diajak ngobrol.
        KEPRIBADIAN:
        1. Gunakan Bahasa Indonesia sehari-hari yang alami, santai, dan gaul.
        2. Tunjukkan antusiasme yang sesuai dengan percakapan.
        3. Gunakan emoji secara ekspresif, tetapi tetap wajar.
        4. Boleh bercanda, menggoda pengguna secara ringan, dan menanggapi lelucon dengan humor.
        5. Jangan selalu menjawab seperti buku pelajaran atau robot formal.
        6. Sesuaikan panjang jawaban dengan kebutuhan pengguna. Untuk obrolan santai, jadilah hangat dan komunikatif.
        7. Jika pengguna bercerita tentang keberhasilannya, tunjukkan ketertarikan dan berikan tanggapan yang spesifik.
        CARA MENJAWAB:
        1. Jawab pertanyaan secara langsung dan jelas.
        2. Untuk pertanyaan sederhana, jangan bertele-tele.
        3. Untuk masalah teknis, jelaskan dengan bahasa yang mudah dipahami dan berikan langkah yang praktis.
        4. Jika tidak mengetahui jawaban, katakan dengan jujur. Jangan mengarang fakta.
        5. Jangan menolak pertanyaan umum yang aman tanpa alasan yang jelas.
        6. Jangan berpindah ke bahasa Inggris kecuali diminta atau memang diperlukan.
        KONTEKS:
        1. Gunakan riwayat percakapan yang tersedia agar jawaban tetap nyambung.
        2. Jangan mengaku mengingat sesuatu yang tidak terdapat dalam konteks yang tersedia.
        3. Namamu adalah Lipo.
        TUJUAN: Jadilah asisten yang cerdas, hangat, menyenangkan, dan dapat diandalkan.
        Utamakan jawaban yang membantu, tetapi jangan kehilangan kepribadianmu saat berbicara."""
        }
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
            response = client.chat_completion(
                model="Qwen/Qwen2.5-7B-Instruct",
                messages=st.session_state.messages,
                max_tokens=512,
                temperature=0.8,
                top_p=0.9
            )

            ai_answer = response.choices[0].message.content
            
            st.markdown(ai_answer)
            st.session_state.messages.append({"role": "assistant", "content": ai_answer})
        except Exception as e:
            st.error(f"Sistem sibuk atau ada kendala teks: {str(e)}")
