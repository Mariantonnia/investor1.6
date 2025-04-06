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
    "Wall Street y los mercados globales caen ante la incertidumbre por la guerra comercial y el temor a una recesión",
    "El mercado de criptomonedas se desploma: Bitcoin cae a 80.000 dólares, las altcoins se hunden en medio de una frenética liquidación",
    "Granada retrasa seis meses el inicio de la Zona de Bajas Emisiones, previsto hasta ahora para abril",
    "McDonald's donará a la Fundación Ronald McDonald todas las ganancias por ventas del Big Mac del 6 de diciembre",
    "El Gobierno autoriza a altos cargos públicos a irse a Indra, Escribano, CEOE, Barceló, Iberdrola o Airbus",
    "Las aportaciones a los planes de pensiones caen 10.000 millones en los últimos cuatro años",
]

plantilla_reaccion = """
Reacción del inversor: {reaccion}
Analiza el sentimiento y la preocupación expresada.  
Clasifica la preocupación principal en una de estas categorías:  
- Ambiental  
- Social  
- Gobernanza  
- Riesgo  

Evalúa si la respuesta es clara y detallada. Debe contener al menos una justificación o explicación. Si solo expresa una opinión sin justificación, devuelve "INSUFICIENTE".

Si la respuesta es insuficiente, genera una pregunta de seguimiento enfocada en la categoría detectada para profundizar en la opinión del inversor, sin imprimir la justificación.
"""
prompt_reaccion = PromptTemplate(template=plantilla_reaccion, input_variables=["reaccion"])
cadena_reaccion = LLMChain(llm=llm, prompt=prompt_reaccion)

if "historial" not in st.session_state:
    st.session_state.historial = []
    st.session_state.contador = 0
    st.session_state.mostrada_noticia = False
    st.session_state.esperando_respuesta = False

st.title("Chatbot de Análisis de Sentimiento")

for mensaje in st.session_state.historial:
    with st.chat_message(mensaje["tipo"]):
        st.write(mensaje["contenido"])

if st.session_state.contador < len(noticias):
    if not st.session_state.mostrada_noticia and not st.session_state.esperando_respuesta:
        noticia = noticias[st.session_state.contador]
        with st.chat_message("bot", avatar="🤖"):
            st.write(f"¿Qué opinas sobre esta noticia? {noticia}")
        st.session_state.historial.append({"tipo": "bot", "contenido": noticia})
        st.session_state.mostrada_noticia = True

    user_input = st.chat_input("Escribe tu respuesta aquí...")
    if user_input:
        st.session_state.historial.append({"tipo": "user", "contenido": user_input})
        
        if st.session_state.esperando_respuesta:
            # Si el usuario está respondiendo a la pregunta de seguimiento, avanzar a la siguiente noticia
            st.session_state.esperando_respuesta = False
            st.session_state.contador += 1
            st.session_state.mostrada_noticia = False
            st.rerun()
        else:
            analisis_reaccion = cadena_reaccion.run(reaccion=user_input)
            
            if "INSUFICIENTE" in analisis_reaccion:
                with st.chat_message("bot", avatar="🤖"):
                    pregunta_seguimiento = analisis_reaccion.replace("INSUFICIENTE", "").strip()
                    st.write(pregunta_seguimiento)
                st.session_state.historial.append({"tipo": "bot", "contenido": pregunta_seguimiento})
                st.session_state.esperando_respuesta = True
            else:
                with st.chat_message("bot", avatar="🤖"):
                    st.write(analisis_reaccion)
                st.session_state.historial.append({"tipo": "bot", "contenido": analisis_reaccion})
                
                # Avanzar a la siguiente noticia si la respuesta es suficiente
                st.session_state.contador += 1
                st.session_state.mostrada_noticia = False
                st.session_state.esperando_respuesta = False
                st.rerun()
else:
    st.write("Análisis completado. Gracias por participar.")
