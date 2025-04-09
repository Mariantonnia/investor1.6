import streamlit as st
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from langchain import LLMChain, PromptTemplate
from langchain_groq import ChatGroq
import os
import re
import json
from dotenv import load_dotenv
import matplotlib.pyplot as plt

load_dotenv()

# Configurar el modelo LLM
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    model="gemma2-9b-it",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

noticias = [
    "Repsol, entre las 50 empresas que más responsabilidad histórica tienen en el calentamiento global",
    "Amancio Ortega crea un fondo de 100 millones de euros para los afectados de la dana",
    "Freshly Cosmetics despide a 52 empleados en Reus, el 18% de la plantilla",
]

# Plantilla para generar solo la pregunta de seguimiento
plantilla_reaccion = """
Reacción del inversor: {reaccion}
Genera ÚNICAMENTE una pregunta de seguimiento enfocada en profundizar en la opinión del inversor.  
Ejemplo:  
"¿Consideras que la existencia de mecanismos robustos de control interno y transparencia podría mitigar tu preocupación por la gobernanza corporativa en esta empresa?"
"""
prompt_reaccion = PromptTemplate(template=plantilla_reaccion, input_variables=["reaccion"])
cadena_reaccion = LLMChain(llm=llm, prompt=prompt_reaccion)

# Función para procesar respuestas válidas con límite de 2 preguntas por noticia
def procesar_respuesta_valida(user_input):
    if "contador_preguntas" not in st.session_state:
        st.session_state.contador_preguntas = 0

    # Generar solo la pregunta de seguimiento
    pregunta_seguimiento = cadena_reaccion.run(reaccion=user_input).strip()
    
    with st.chat_message("bot", avatar="🤖"):
        if st.session_state.contador_preguntas < 2:
            if "¿" in pregunta_seguimiento:  # Verificar que sea una pregunta válida
                st.write(pregunta_seguimiento)
                st.session_state.pregunta_pendiente = pregunta_seguimiento
                st.session_state.contador_preguntas += 1
            else:
                st.write("Gracias por tus respuestas. Avanzando a la siguiente noticia...")
                st.session_state.contador += 1
                st.session_state.mostrada_noticia = False
                st.session_state.contador_preguntas = 0  # Resetear contador
        else:
            st.write("Gracias por tus respuestas. Avanzando a la siguiente noticia...")
            st.session_state.contador += 1
            st.session_state.mostrada_noticia = False
            st.session_state.contador_preguntas = 0
    
    st.session_state.historial.append({"tipo": "bot", "contenido": pregunta_seguimiento})
    st.session_state.reacciones.append(user_input)
    st.rerun()

# Inicialización de estados
if "historial" not in st.session_state:
    st.session_state.historial = []
    st.session_state.contador = 0
    st.session_state.reacciones = []
    st.session_state.mostrada_noticia = False
    st.session_state.contador_preguntas = 0

st.title("Chatbot de Análisis de Sentimiento")

# Mostrar historial del chat
for mensaje in st.session_state.historial:
    with st.chat_message(mensaje["tipo"]):
        st.write(mensaje["contenido"])

# Lógica principal del chat
if st.session_state.contador < len(noticias):
    if not st.session_state.mostrada_noticia:
        noticia = noticias[st.session_state.contador]
        with st.chat_message("bot", avatar="🤖"):
            st.write(f"¿Qué opinas sobre esta noticia? {noticia}")
        st.session_state.historial.append({"tipo": "bot", "contenido": noticia})
        st.session_state.mostrada_noticia = True

    user_input = st.chat_input("Escribe tu respuesta aquí...")
    if user_input:
        st.session_state.historial.append({"tipo": "user", "contenido": user_input})
        
        if hasattr(st.session_state, 'pregunta_pendiente'):
            del st.session_state.pregunta_pendiente
            procesar_respuesta_valida(user_input)
        else:
            procesar_respuesta_valida(user_input)
else:
    # Generación del perfil final (simplificado)
    with st.chat_message("bot", avatar="🤖"):
        perfil_final = "\n".join(st.session_state.reacciones)
        st.write(f"**Perfil del inversor basado en sus respuestas:**\n{perfil_final}")
