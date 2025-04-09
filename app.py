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

plantilla_reaccion = """
Reacción del inversor: {reaccion}
Analiza el sentimiento y la preocupación expresada.  
Clasifica la preocupación principal en una de estas categorías:
- Ambiental
- Social
- Riesgo

Si la respuesta es demasiado breve o poco clara, solicita más detalles de manera específica.  
Luego, genera una pregunta de seguimiento enfocada en la categoría detectada.
"""
prompt_reaccion = PromptTemplate(template=plantilla_reaccion, input_variables=["reaccion"])
cadena_reaccion = LLMChain(llm=llm, prompt=prompt_reaccion)

plantilla_perfil = """
Análisis de reacciones: {analisis}
Genera un perfil detallado del inversor basado en sus reacciones, enfocándote en los pilares ESG (Ambiental, Social) y su aversión al riesgo.
Asigna una puntuación de 0 a 100 para cada pilar ESG y para el riesgo.
Devuelve las puntuaciones en formato: Ambiental: [puntuación], Social: [puntuación], Riesgo: [puntuación]
"""
prompt_perfil = PromptTemplate(template=plantilla_perfil, input_variables=["analisis"])
cadena_perfil = LLMChain(llm=llm, prompt=prompt_perfil)

if "historial" not in st.session_state:
    st.session_state.historial = []
    st.session_state.contador = 0
    st.session_state.reacciones = []
    st.session_state.mostrada_noticia = False
    st.session_state.pregunta_clarificacion = False

st.title("Chatbot de Análisis de Sentimiento")

for mensaje in st.session_state.historial:
    with st.chat_message(mensaje["tipo"]):
        st.write(mensaje["contenido"])

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
        
        if not st.session_state.pregunta_clarificacion:
            analisis_reaccion = cadena_reaccion.run(reaccion=user_input)
            if "¿" in analisis_reaccion:  # Si generó una pregunta, hacer solo una vez
                with st.chat_message("bot", avatar="🤖"):
                    st.write(analisis_reaccion)
                st.session_state.historial.append({"tipo": "bot", "contenido": analisis_reaccion})
                st.session_state.pregunta_clarificacion = True
            else:
                with st.chat_message("bot", avatar="🤖"):
                    st.write("Ok, pasemos a la siguiente pregunta.")
                st.session_state.contador += 1
                st.session_state.mostrada_noticia = False
                st.session_state.pregunta_clarificacion = False
                st.rerun()
        else:
            with st.chat_message("bot", avatar="🤖"):
                st.write("Ok, pasemos a la siguiente pregunta.")
            st.session_state.contador += 1
            st.session_state.mostrada_noticia = False
            st.session_state.pregunta_clarificacion = False
            st.rerun()
else:
    analisis_total = "\n".join(st.session_state.reacciones)
    perfil = cadena_perfil.run(analisis=analisis_total)
    
    puntuaciones = {"Ambiental": 0, "Social": 0, "Riesgo": 0}
    for clave in puntuaciones.keys():
        match = re.search(fr"{clave}: (\d+)", perfil)
        if match:
            puntuaciones[clave] = int(match.group(1))
    
    with st.chat_message("bot", avatar="🤖"):
        st.write(f"**Perfil del inversor:** {perfil}")
    st.session_state.historial.append({"tipo": "bot", "contenido": f"**Perfil del inversor:** {perfil}"})

    fig, ax = plt.subplots()
    ax.bar(puntuaciones.keys(), puntuaciones.values())
    ax.set_ylabel("Puntuación (0-100)")
    ax.set_title("Perfil del Inversor")
    st.pyplot(fig)

    try:
        creds_json_str = st.secrets["gcp_service_account"]
        creds_json = json.loads(creds_json_str)
    except Exception as e:
        st.error(f"Error al cargar las credenciales: {e}")
        st.stop()
    
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_json, scope)
    client = gspread.authorize(creds)
    sheet = client.open('BBDD_RESPUESTAS').sheet1
    
    fila = st.session_state.reacciones[:]
    fila.extend(puntuaciones.values())
    sheet.append_row(fila)
    
    st.success("Respuestas y perfil guardados en Google Sheets en una misma fila.")
