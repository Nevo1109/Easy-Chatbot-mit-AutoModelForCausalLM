import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

@st.cache_resource
def load_model():
    model_name = "microsoft/DialoGPT-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

st.title("Mini Chatbot Demo")
st.write("Ein einfacher Chatbot mit DialoGPT")

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("Du:", key="input")

if user_input:
    # Eingabe kodieren
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    # Vorherigen Verlauf anhängen (wenn vorhanden)
    if st.session_state.history:
        bot_input_ids = torch.cat([st.session_state.history[-1], new_input_ids], dim=-1)
    else:
        bot_input_ids = new_input_ids

    # Antwort generieren
    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.8
    )

    # Antwort extrahieren
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    # Verlauf speichern
    st.session_state.history.append(chat_history_ids)

    # Ausgabe
    st.text_area("Bot:", value=response, height=100)