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

# Cargar variables de entorno
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

# Prompt para evaluar si la respuesta es suficiente
prompt_evaluacion = PromptTemplate(
    template="""
    Respuesta del inversor: {respuesta}
    Evalúa si la respuesta es suficientemente detallada o si es vaga e inespecífica.
    Si la respuesta es demasiado genérica, devuelve: "Suficiente: No"
    Si la respuesta es clara y bien fundamentada, devuelve: "Suficiente: Sí"
    """,
    input_variables=["respuesta"]
)
cadena_evaluacion = LLMChain(llm=llm, prompt=prompt_evaluacion)

# Prompt para generar preguntas de seguimiento
prompt_reaccion = PromptTemplate(
    template="""
    Reacción del inversor: {reaccion}
    Si la respuesta es vaga, genera una pregunta de seguimiento para obtener más detalles.
    Si la respuesta ya es clara, devuelve: "No es necesario más detalle".
    """,
    input_variables=["reaccion"]
)
cadena_reaccion = LLMChain(llm=llm, prompt=prompt_reaccion)

# Inicializar estado en Streamlit
if "historial" not in st.session_state:
    st.session_state.historial = []
    st.session_state.contador = 0
    st.session_state.reacciones = []
    st.session_state.mostrada_noticia = False
    st.session_state.esperando_respuesta_extra = False  # Nuevo flag

st.title("Chatbot de Análisis de Sentimiento")

# Mostrar historial
for mensaje in st.session_state.historial:
    with st.chat_message(mensaje["tipo"]):
        st.write(mensaje["contenido"])

# Mostrar noticia y pedir reacción
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
        st.session_state.reacciones.append(user_input)

        if st.session_state.esperando_respuesta_extra:
            # Si el usuario responde a la pregunta de seguimiento, avanzamos de noticia
            st.session_state.contador += 1
            st.session_state.mostrada_noticia = False
            st.session_state.esperando_respuesta_extra = False
            st.rerun()
        else:
            # Evaluamos si la respuesta es suficientemente detallada
            evaluacion = cadena_evaluacion.run(respuesta=user_input)
            suficiente_match = re.search(r"Suficiente: (Sí|No)", evaluacion)

            if suficiente_match and suficiente_match.group(1) == "No":
                # Generar una pregunta de seguimiento
                pregunta_seguimiento = cadena_reaccion.run(reaccion=user_input)
                if "No es necesario más detalle" not in pregunta_seguimiento:
                    st.session_state.esperando_respuesta_extra = True
                    with st.chat_message("bot", avatar="🤖"):
                        st.write(pregunta_seguimiento)
                    st.session_state.historial.append({"tipo": "bot", "contenido": pregunta_seguimiento})
                else:
                    # Si no es necesario más detalles, pasamos a la siguiente noticia
                    st.session_state.contador += 1
                    st.session_state.mostrada_noticia = False
                    st.rerun()
            else:
                # Si la respuesta es suficiente, pasamos directamente a la siguiente noticia
                st.session_state.contador += 1
                st.session_state.mostrada_noticia = False
                st.rerun()
else:
    with st.chat_message("bot", avatar="🤖"):
        st.write("Fin del análisis. ¡Gracias por participar!")
    st.session_state.historial.append({"tipo": "bot", "contenido": "Fin del análisis. ¡Gracias por participar!"})
