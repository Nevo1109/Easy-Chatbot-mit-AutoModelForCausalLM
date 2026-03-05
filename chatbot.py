from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Modell und Tokenizer laden
model_name = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Chat-Verlauf speichern
chat_history_ids = None

print("Chatbot ist bereit. Beenden mit 'exit'.")

while True:
    user_input = input("Du: ")
    if user_input.lower() == "exit":
        break

    # Eingabe tokenisieren
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    # Verlauf anhängen
    bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1) if chat_history_ids is not None else new_input_ids

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

    # Antwort dekodieren und ausgeben
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    print(f"Bot: {response}")