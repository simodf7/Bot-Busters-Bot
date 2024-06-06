import base64
import streamlit as st
import os
import asyncio
from setup import prepare_rag_llm, generate_answer  # Si assume che queste funzioni siano asincrone
import torch
import time
import gc
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

# Funzioni per la gestione della memoria
def clear_gpu_memory():
    # Libera la memoria della GPU
    torch.cuda.empty_cache()
    gc.collect()

def wait_until_enough_gpu_memory(min_memory_available, max_retries=10, sleep_time=5):
    # Inizializza NVML (NVIDIA Management Library)
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(torch.cuda.current_device())

    # Tenta di liberare abbastanza memoria GPU con un massimo di tentativi
    for _ in range(max_retries):
        info = nvmlDeviceGetMemoryInfo(handle)
        if info.free >= min_memory_available:
            break
        print(f"Waiting for {min_memory_available} bytes of free GPU memory. Retrying in {sleep_time} seconds...")
        time.sleep(sleep_time)
    else:
        # Solleva un'eccezione se non si riesce a liberare abbastanza memoria
        raise RuntimeError(f"Failed to acquire {min_memory_available} bytes of free GPU memory after {max_retries} retries.")

# Funzione principale asincrona
async def main():
    # Chiamata alle funzioni di gestione della memoria prima di avviare l'app Streamlit
    clear_gpu_memory()

    st.set_page_config(page_title="Bot Busters Bot",
                       page_icon=":robot_face:",
                       layout="wide")

    # Inizializza la lista di dizionari conversations nel dizionario session_state di Streamlit
    # Questi dizionari conterranno:
    # - id della conversazione
    # - domande e risposte effettuate precedentemente nella conversazione
    
    # Definiamo anche la variabile current chat id per tenere traccia degli id delle conversazioni
    # La prima volta e' settata a 1
    if "conversations" not in st.session_state:
        st.session_state.conversations = []
        st.session_state.current_chat_id = 1
        first_conv = {'id': st.session_state.current_chat_id, 'questions': [], 'answers': []}
        st.session_state.conversations.append(first_conv)

    # Pulsante per avviare una nuova conversazione
    # quando creiamo una nuova conversazione incrementiamo l'id e aggiungiamo una nuova chat a conversations
    # rendendola poi la chat attiva correntemente settando opportunamente la variabile current_chat con l'id della conversazione
    if st.sidebar.button(":speech_balloon: Nuova conversazione"):
        st.session_state.current_chat_id += 1
        new_chat = {"id": st.session_state.current_chat_id, 'questions': [], 'answers': []}
        st.session_state.conversations.append(new_chat)
        st.session_state.history = []  # Resetta la cronologia per la nuova conversazione
        st.session_state.current_chat = st.session_state.current_chat_id

    st.sidebar.title("Chat Precedenti")

    # Visualizza le chat precedenti nella barra laterale
    # inizialmente quando creiamo la conversazione il nome visualizzato nella sidebar sara' Chat + il suo id
    # non appena poniamo la prima domanda verra' sostituito il nome con la prima domanda effettuata

    # quando clicchiamo una delle conversazione precedenti la lista history sara' ooportunamente aggiornata 
    # con le interazioni precedenti ricavabili dalle liste question and answer all'interno del dizionario conversations
    for chat in st.session_state.conversations:
        if st.sidebar.button(chat["questions"][0] if chat["questions"] else f"Chat {chat['id']}"):
            st.session_state.current_chat = chat["id"]
            st.session_state.history = []
            for i in range(len(chat['questions'])):
                st.session_state.history.append({"role": "user", "content": chat["questions"][i]})
                st.session_state.history.append({"role": "assistant", "content": chat["answers"][i]})

    # Visualizza la pagina del chatbot
    await display_chatbot_page()

# Funzione per caricare e codificare un'immagine in base64
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Funzione asincrona per visualizzare la pagina del chatbot
async def display_chatbot_page():
    st.markdown("# BotBusters Bot")

    # Percorso dell'immagine locale
    background_image = 'dark.png'
    bg_image_base64 = get_base64_of_bin_file(background_image)

    # Codice CSS per un'immagine di background fissa, più sfumata e ridimensionata
    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] > .main {{
        display: flex;
        justify-content: center;
        align-items: center;
        background: url("data:image/jpeg;base64,{bg_image_base64}");
        background-size: 600px 600px;  /* Ridimensiona l'immagine */
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: inherit;
    }}
    </style>
    """
    # Applica il CSS
    st.markdown(page_bg_img, unsafe_allow_html=True)

    # Stato della sessione per la chat corrente
    if "current_chat" not in st.session_state:
        st.session_state.current_chat = 1

    # settiamo la entry agent nel dizionario session state di agent
    # sara' poi settata con l'agente dichiarato poi con prepare_rag_llm
    if "agent" not in st.session_state:
        st.session_state.agent = None

    existing_vector_store = "parlamento"
    temperature = 0.5
    max_length = 500

    # Prepara l'agente LLM
    st.session_state.agent = await prepare_rag_llm(  # Attende la preparazione del modello LLM
        existing_vector_store, temperature, max_length
    )

    # Cronologia della chat
    if "history" not in st.session_state:
        st.session_state.history = []

    # Visualizza la sequenza di messaggi inviati precedentemente nella schermata della conversazione corrente
    # questo grazie alla history ottenuta precedentemente
    for message in st.session_state.history:
        with st.chat_message(message["role"]):
            if message["role"] == 'assistant':
                st.markdown(message["content"]["content"])
                # Aggiungi un menu a discesa per visualizzare le fonti
                 # Aggiungi un menu a discesa per visualizzare le fonti
                with st.expander("Source"):
                    i = 1
                    for doc in message["content"]["sources"]:
                        st.markdown(f"Fonte {i}: {doc}")
                        i +=1
            else:
                st.markdown(message["content"])



    # Come abbiamo detto prima quando cambiamo chat, l'indice della current chat cambia
    # Questo e' utile perche dobbiamo ottenere il dizionario corrispondente alla conversazione per salvare nuove domande e nuove risposte
    if question := st.chat_input("Fammi una domanda!"):
        current_chat = next((chat for chat in st.session_state.conversations if chat["id"] == st.session_state.current_chat), None)

        # Aggiunge la domanda dell'utente al dizionario della chat
        current_chat["questions"].append(question)

        # Visualizza la domanda dell'utente
        with st.chat_message("user"):
            st.markdown(question)

        # Genera la risposta alla domanda asincronamente
        answer,source = await generate_answer(st.session_state.agent, question)  # Attende la generazione della risposta
        with st.chat_message("assistant"):
            st.markdown(answer)
            with st.expander("Source"):
                i = 1
                for doc in source:
                    st.markdown(f"Fonte {i}: {doc}")
                    i +=1

        # Aggiunge la risposta del modello al dizionario della chat
        current_chat["answers"].append({'content': answer, 'sources': source})


if __name__ == "__main__":
    asyncio.run(main())
