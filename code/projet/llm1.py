import streamlit as st
import json
import os
from ingest import initialize_vector_store
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings.ollama import OllamaEmbeddings
from docling_east import process_image_with_east_docling

# Configuration de l'interface
st.set_page_config(
    page_title="TextraHealth - Analyseur d'Ordonnances",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://example.com',
        'About': "Application d'analyse des ordonnances médicales"
    }
)

# Styles CSS personnalisés
st.markdown("""
<style>
    :root {
        --primary: #1E88E5;
        --secondary: #0D47A1;
        --accent: #43A047;
        --warning: #d32f2f;
    }
    
    .main-title {
        font-size: 2.5rem;
        color: var(--primary);
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    
    .tab-container {
        border-radius: 10px;
        padding: 1rem;
        margin-top: 1rem;
        background: #f8f9fa;
    }
    
    .medication-card {
        border-left: 4px solid var(--primary);
        padding: 1rem;
        margin-bottom: 1rem;
        background: white;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .chat-message-user {
        background-color: var(--primary);
        color: white;
        border-radius: 15px 15px 0 15px;
        padding: 10px;
        margin: 5px 0;
        max-width: 80%;
        margin-left: auto;
    }

    .chat-message-bot {
        background-color: #e0e0e0; /* gris plus foncé pour contraste */
        color: #000000; /* texte noir */
        border-radius: 15px 15px 15px 0;
        padding: 10px;
        margin: 5px 0;
        max-width: 80%;
        margin-right: auto;
    }

    @media (prefers-color-scheme: dark) {
        .chat-message-bot {
            background-color: #2c2c2c; /* plus foncé pour dark mode */
            color: #f1f1f1; /* texte clair */
        }
    }
    
    .stButton>button {
        background-color: var(--primary);
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    
    .stButton>button:hover {
        background-color: var(--secondary);
    }
</style>
""", unsafe_allow_html=True)

# Gestion de l'état
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text = None
if 'report_text' not in st.session_state:
    st.session_state.report_text = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Fonctions principales
def extract_text_from_prescription(file):
    try:
        temp_path = f"temp_{file.name}"
        with open(temp_path, "wb") as f:
            f.write(file.getbuffer())
        
        extracted_lines = process_image_with_east_docling(temp_path)
        os.remove(temp_path)
        return "\n".join(extracted_lines)
    except Exception as e:
        st.error(f"Erreur d'extraction : {str(e)}")
        return None

def generate_medical_report(extracted_text):
    model = ChatOllama(model="llama3.2:3b")
    
    prompt = """Vous êtes un llm qui va étre un assistant médical expert.Vous aurez comme entrée la sortie d'un modèle OCR qui va extraire le texte d'une ordonnnace médicale, et ce texte sera surement trop bruité et incorrecte.Vous devez analyser ce texte fourni par ocr, voici votre tàche :

-Corrigez les erreurs des noms des médicaments extrait par ocr et les erreurs de dosage (ex "20o mg" doit étre 200 mg" et corrige toute les erreurs que vous trouvez sans mentionner qu'il y avait une erreur, montre la sortie au utilisateur bien claire et bien expliqué.


📋 **Médicaments Prescrits**:
- [Nom] (Dosage: X, Fréquence: Y, Durée: Z)

💡 **Conseils Importants**:
- Prendre avec de l'eau
- Éviter l'alcool

⚠️ **Avertissements**:
- Risque de somnolence
- Contre-indications

Texte: {prescription_text}"""
    
    chain = (
        {"prescription_text": RunnablePassthrough()}
        | ChatPromptTemplate.from_template(prompt)
        | model
        | StrOutputParser()
    )
    
    return chain.invoke(extracted_text)

def chat_with_llm(question, context=""):
    model = ChatOllama(model="llama3.2:3b")
    db = initialize_vector_store()
    prompt = """Vous êtes un assistant virtuel qui doit répondre aux questions des gens concernant leurs ordonnnaces. Répondez à cette question sur les médicaments:
    
Contexte de l'ordonnance:
{context}

Question: {question}

Répondez de manière claire et précise en français."""

    chain = (
        {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
        | ChatPromptTemplate.from_template(prompt)
        | model
        | StrOutputParser()
    )
    
    return chain.invoke({"context": context, "question": question})

# Barre latérale
with st.sidebar:
    st.image("logo.jpeg", width=150)
    st.markdown("## TextraHealth")
    st.markdown("""
    **Fonctionnalités:**
    - Analyse intelligente d'ordonnances
    - Rapport détaillé sur les médicaments prescrits(dosage, fréquences d'utilisation, effets secondaires...)
    - Assistant virtuel
    """)
    st.markdown("---")
    st.markdown("ℹ️ **Avertissement:** Cet outil ne remplace pas un avis médical professionnel.")

# Page principale
st.markdown("<h1 class='main-title'>TextraHealth</h1>", unsafe_allow_html=True)
st.markdown("### Votre assistant intelligent pour les ordonnances médicales")

# Onglets
tab1, tab2 = st.tabs(["📄 Analyse d'Ordonnance", "💬 Chatbot Médical"])

with tab1:
    st.markdown("#### Téléchargez votre ordonnance")
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    
    if uploaded_file:
        st.session_state.uploaded_file = uploaded_file
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image(uploaded_file, caption="Votre ordonnance")
        
        with col2:
            if st.button("Analyser l'ordonnance", use_container_width=True):
                with st.spinner("Analyse en cours..."):
                    extracted_text = extract_text_from_prescription(uploaded_file)
                    if extracted_text:
                        st.session_state.extracted_text = extracted_text
                        report_text = generate_medical_report(extracted_text)
                        if report_text:
                            st.session_state.report_text = report_text
                            st.success("Analyse terminée!")
    
    if st.session_state.get('report_text'):
        st.markdown("---")
        st.markdown("## 📋 Rapport d'Analyse")
        st.markdown(st.session_state.report_text)

with tab2:
    st.markdown("#### Posez vos questions sur les médicaments")
    
    if st.session_state.get('extracted_text'):
        st.info("💡 Le chatbot a accès au contenu de votre ordonnance pour des réponses personnalisées")
    
    user_input = st.text_input("Votre question:", placeholder="Ex: Quels sont les effets secondaires de ce médicament?", label_visibility="collapsed")
    
    if st.button("Envoyer", key="send_chat"):
        if user_input:
            context = st.session_state.get('extracted_text', "")
            with st.spinner("Recherche de la réponse..."):
                response = chat_with_llm(user_input, context)
                st.session_state.chat_history.append(("user", user_input))
                st.session_state.chat_history.append(("bot", response))
    
    st.markdown("---")
    
    for sender, message in st.session_state.chat_history:
        if sender == "user":
            st.markdown(f'<div class="chat-message-user">{message}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message-bot">{message}</div>', unsafe_allow_html=True)

# Pied de page
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    © 2025 TextraHealth | Analyseur des ordonnances
</div>
""", unsafe_allow_html=True)