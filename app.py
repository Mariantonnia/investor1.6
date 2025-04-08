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

Evalúa si la respuesta es clara y detallada. Debe contener al menos una justificación o explicación. Si solo expresa una opinión sin justificación o es demasiado corta (menos de 5 palabras), devuelve "INSUFICIENTE".

Si la respuesta es insuficiente, genera una pregunta de seguimiento enfocada en la categoría detectada para profundizar en la opinión del inversor. Devuelve SOLO LA PREGUNTA, sin ninguna explicación adicional.
"""

prompt_reaccion = PromptTemplate(template=plantilla_reaccion, input_variables=["reaccion"])
cadena_reaccion = LLMChain(llm=llm, prompt=prompt_reaccion)

plantilla_perfil = """
Análisis de reacciones: {analisis}

Genera un perfil detallado del inversor basado en sus reacciones, enfocándote en los pilares ESG (Ambiental, Social y Gobernanza) y su aversión al riesgo.

Asigna una puntuación de 0 a 100 para cada pilar ESG y para el riesgo, donde 0 indica ninguna preocupación y 100 máxima preocupación o aversión.

Devuelve las 4 puntuaciones en formato: Ambiental: [puntuación], Social: [puntuación], Gobernanza: [puntuación], Riesgo: [puntuación]
"""

prompt_perfil = PromptTemplate(template=plantilla_perfil, input_variables=["analisis"])
cadena_perfil = LLMChain(llm=llm, prompt=prompt_perfil)

if "historial" not in st.session_state:
    st.session_state.historial = []
    st.session_state.contador = 0
    st.session_state.mostrada_noticia = False
    st.session_state.esperando_respuesta = False
    st.session_state.reacciones = []
    st.session_state.ultima_pregunta = ""

st.title("Chatbot de Análisis de Sentimiento")

for mensaje in st.session_state.historial:
    with st.chat_message(mensaje["tipo"]):
        st.write(mensaje["contenido"])

if st.session_state.contador < len(noticias):
    if not st.session_state.esperando_respuesta:
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
        
        if st.session_state.esperando_respuesta:
            st.session_state.esperando_respuesta = False
            st.session_state.ultima_pregunta = ""
            st.session_state.contador += 1
            st.session_state.mostrada_noticia = False
            st.rerun()
        
        analisis_reaccion = cadena_reaccion.run(reaccion=user_input)
        
        if analisis_reaccion.startswith("¿"):
            with st.chat_message("bot", avatar="🤖"):
                st.write(analisis_reaccion)
            st.session_state.historial.append({"tipo": "bot", "contenido": analisis_reaccion})
            st.session_state.esperando_respuesta = True
            st.session_state.ultima_pregunta = analisis_reaccion
        else:
            with st.chat_message("bot", avatar="🤖"):
                st.write(f"La preocupación principal es {analisis_reaccion}.")
            st.session_state.historial.append({"tipo": "bot", "contenido": f"La preocupación principal es {analisis_reaccion}."})
            st.session_state.contador += 1
            st.session_state.mostrada_noticia = False
            st.session_state.esperando_respuesta = False
            st.rerun()
else:
    analisis_total = "\n".join(st.session_state.reacciones)
    perfil = cadena_perfil.run(analisis=analisis_total)
    with st.chat_message("bot", avatar="🤖"):
        st.write(f"**Perfil del inversor:** {perfil}")
    st.session_state.historial.append({"tipo": "bot", "contenido": f"**Perfil del inversor:** {perfil}"})
    
    puntuaciones = {k: int(re.search(fr"{k}: (\d+)", perfil).group(1)) for k in ["Ambiental", "Social", "Gobernanza", "Riesgo"]}
    
    fig, ax = plt.subplots()
    ax.bar(puntuaciones.keys(), puntuaciones.values())
    ax.set_ylabel("Puntuación (0-100)")
    ax.set_title("Perfil del Inversor")
    st.pyplot(fig)
    
    try:
        creds_json = json.loads(st.secrets["gcp_service_account"])
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        client = gspread.authorize(ServiceAccountCredentials.from_json_keyfile_dict(creds_json, scope))
        sheet = client.open('BBDD_RESPUESTAS').sheet1
        sheet.append_row(st.session_state.reacciones + list(puntuaciones.values()))
        st.success("Respuestas y perfil guardados en Google Sheets.")
    except Exception as e:
        st.error(f"Error al guardar los datos: {e}")

