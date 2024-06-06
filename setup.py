import json
import openai
import streamlit as st
from pypdf import PdfReader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.agents import AgentExecutor, Tool
from langchain.memory import ConversationBufferWindowMemory
from langchain.vectorstores.faiss import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.prompts.chat import MessagesPlaceholder, ChatPromptTemplate
from langchain.agents.openai_tools.base import create_openai_tools_agent
from langchain_openai.chat_models.base import ChatOpenAI
import os
import asyncio


# Imposta una variabile d'ambiente per evitare errori con OpenMP
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Imposta la chiave API di OpenAI dalle variabili segrete di Streamlit
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Funzione asincrona per leggere un file PDF e estrarre il testo
async def read_pdf(file):
    """
    Legge un file PDF e restituisce il testo estratto.
    """
    document = ""
    reader = PdfReader(file)
    for page in reader.pages:
        document += page.extract_text()
    return document

# Funzione asincrona per leggere un file di testo e pulire i caratteri speciali
async def read_txt(file):
    """
    Legge un file di testo e lo pre-processa.
    """
    document = str(file.getvalue())
    document = document.replace("\\n", " \\n ").replace("\\r", " \\r ")
    return document

# Funzione per dividere un documento in chunk 
def split_doc(document, chunk_size, chunk_overlap):
    """
    Divide un documento in chunk di testo più piccoli tramite l'utilizzo di RCTS di Langchain.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,  # Dimensione massima di ogni frammento
        chunk_overlap=chunk_overlap  # Sovrapposizione tra frammenti
    )
    split = splitter.split_text(document)  # Divide il testo
    split = splitter.create_documents(split)  # Crea documenti dai frammenti
    return split

# Funzione asincrona per memorizzare embeddings e unire database esistenti
async def embedding_storing(split, existing_vector_store):
    """
    Memorizza embeddings dei chunk e unisce questi embeddings a un database esistente sfruttando l'indicizzazione di Faiss.
    """
    # Imposto il modello di Embedding di OpenAI
    instructor_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


    # Crea un'istanza di FAISS con gli argomenti richiesti
    # Gli embeddings di OpenAI sono gia' normalizzati 
    # Essendo la metrica di similarita' consigliata da OpenAI la Cosine Similarity
    # basta impostare come distance strategy di Faiss il prodotto scalare. 
    faiss = FAISS(
        embedding_function=instructor_embeddings, 
        index=None,  # Indice FAISS appropriato
        docstore=None,  # Docstore appropriato
        index_to_docstore_id=None,  # Mappa indice-docstore appropriata
        distance_strategy=DistanceStrategy.DOT_PRODUCT
    )


    # Ricavo embeddings dai documenti caricati
    db = faiss.from_documents(split, instructor_embeddings)

    # Carica il database esistente
    load_db = faiss.load_local(
        "vector_store/" + existing_vector_store,
        instructor_embeddings,
        allow_dangerous_deserialization=True,
    )
    # Unisci i due database e salva
    load_db.merge_from(db)
    load_db.save_local("vector_store/" + existing_vector_store)

# Funzione asincrona per preparare un modello di linguaggio RAG
async def prepare_rag_llm(vector_store_list, temperature, max_length):
    """
    Prepara un modello di linguaggio RAG (Retrieval-Augmented Generation) utilizzando un database di embeddings.
    Si dichiara un agente conversazionale. 
    """
    # Carica embeddings utilizzando OpenAI
    instructor_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Crea un'istanza di FAISS con gli argomenti richiesti
    faiss = FAISS(
        embedding_function=instructor_embeddings, 
        index=None,  # Indice FAISS appropriato
        docstore=None,  # Docstore appropriato
        index_to_docstore_id=None,  # Mappa indice-docstore appropriata
        distance_strategy=DistanceStrategy.DOT_PRODUCT
    )

    # Carica il database
    loaded_db = faiss.load_local(
        f"vector_store/{vector_store_list}", instructor_embeddings, allow_dangerous_deserialization=True
    )

    # Carica il modello di linguaggio utilizzando OpenAI
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=temperature,
        max_tokens=max_length
    )

    # Configura la memoria conversazionale
    memory = ConversationBufferWindowMemory(
        k=5,  # Numero di messaggi da mantenere in memoria
        return_messages=True,
    )

    # Crea il toolkit degli strumenti
    toolkit = [
        Tool(
            name="DocumentRetriever",
            func=loaded_db.as_retriever(search_kwargs={"k": 5}).get_relevant_documents,
            description="Utile per recuperare documenti relativi alle domande degli utenti."
        ),
        Tool(
            name="DocumentRetrieverJSON",
            func=lambda query: json.dumps(
                [
                    {"page_content": doc.page_content, "metadata": doc.metadata}
                    for doc in loaded_db.as_retriever(search_kwargs={"k": 5}).get_relevant_documents(query)
                ]
            ),
            description="Restituisce i documenti rilevanti per una query in formato JSON."
        )
    ]

    # Definisce il prompt per l'agente
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """
              Sei un assistente legale AI altamente competente ed efficiente, specializzato nel recupero e nell'analisi di documenti legali per fornire risposte accurate e approfondite a domande legali complesse. I tuoi compiti includono:

                1. **Recupero Documenti**:
                - Cerca e recupera informazioni rilevanti da un set di documenti legali forniti.
                - Assicurati che il contenuto recuperato sia direttamente correlato alla query.

                2. **Generazione di Risposte**:
                - Usa il contenuto recuperato per formulare risposte complete e accurate a domande legali.
                - Quando rispondi alle domande, assicurati che le tue risposte siano chiare, concise e aderiscano agli standard e alle terminologie legali.
                - Fornisci citazioni o riferimenti ai documenti utilizzati per generare le tue risposte.

                3. **Gestione di Domande Complesse**:
                - Rispondi a domande legali complesse con analisi approfondite e spiegazioni dettagliate.
                - Suddividi concetti e terminologie legali complessi in parti comprensibili mantenendo l'integrità delle informazioni legali.

                4. **Consapevolezza del Contesto**:
                - Mantieni una comprensione del contesto e delle specifiche di ogni query per adattare le tue risposte con precisione.
                - Tieni traccia delle interazioni precedenti e dei documenti di riferimento per fornire follow-up coerenti e pertinenti.

                Le tue risposte dovrebbero sempre essere professionali, accurate e utili, riflettendo le migliori pratiche di un diligente assistente legale. Ecco alcuni esempi di come dovresti rispondere:

                **Esempio 1**:
                *Domanda*: "Quali sono le principali considerazioni in un caso di violazione del contratto?"
                *Risposta*: "In un caso di violazione del contratto, le principali considerazioni includono l'esistenza di un contratto valido, i termini specifici e gli obblighi delineati nel contratto, la natura della violazione e i danni risultanti. Anche la giurisprudenza e le leggi pertinenti giocano un ruolo critico. Ad esempio, nel caso di [Nome del Caso], il tribunale ha stabilito che..."

                **Esempio 2**:
                *Domanda*: "Puoi riassumere i punti principali del Data Privacy Act?"
                *Risposta*: "Il Data Privacy Act si concentra sulla protezione dei dati personali degli individui regolando come le organizzazioni raccolgono, conservano e utilizzano queste informazioni. Le disposizioni chiave includono il requisito del consenso informato, le misure di sicurezza dei dati e i diritti degli individui di accedere e correggere i propri dati. Ad esempio, la Sezione 3 dell'Atto stipula che..."

                **Esempio 3**:
                *Domanda*: "Come si applica il Fair Use Doctrine ai materiali educativi?"
                *Risposta*: "Il Fair Use Doctrine consente l'uso di materiale protetto da copyright senza permesso in determinate condizioni, come per scopi educativi. I fattori considerati includono lo scopo e il carattere dell'uso, la natura dell'opera protetta da copyright, la quantità utilizzata e l'effetto sul valore di mercato. Secondo [Documento/Caso], il fair use nell'educazione spesso coinvolge..."

                Si prega di procedere rispondendo alla seguente domanda o recuperando i documenti rilevanti secondo necessità.
              """),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    # Crea l'agente utilizzando il toolkit e il prompt definiti
    agent = create_openai_tools_agent(
        llm=llm,
        tools=toolkit,
        prompt=prompt
    )

    # Configura l'esecutore dell'agente
    agent_executor = AgentExecutor(
        agent=agent,
        tools=toolkit,
        memory=memory,
        verbose=True
    )

    return agent_executor

# Funzione asincrona per generare una risposta utilizzando l'agente
async def generate_answer(agent_executor, question):
    """
    Genera una risposta a una domanda utilizzando l'agente AI configurato.
    """
    response = await agent_executor.ainvoke({"input": question})
    answer = response["output"].strip() if "output" in response else "Nessuna risposta generata."
    
    # Utilizzare lo strumento DocumentRetrieverJSON per ottenere i documenti di origine in formato JSON
    doc_json = agent_executor.tools[1].func(question)
    docs = json.loads(doc_json)

    source = []
   

    for doc in docs:
        source.append(doc['page_content'])

    print(source)

    return answer,source
