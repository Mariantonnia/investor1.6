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

# Prompt para analizar la reacción del inversor
prompt_reaccion = PromptTemplate(
    template="""
    Respuesta del inversor: {reaccion}
    Analiza el sentimiento y la preocupación expresada sobre la noticia.
    """,
    input_variables=["reaccion"]
)
cadena_reaccion = LLMChain(llm=llm, prompt=prompt_reaccion)

# Prompt para evaluar si la respuesta es suficiente
prompt_evaluacion = PromptTemplate(
    template="""
    Respuesta del inversor: {respuesta}
    
    Indica únicamente una de las siguientes opciones:
    - "Suficiente: Sí"
    - "Suficiente: No"
    """,
    input_variables=["respuesta"]
)
cadena_evaluacion = LLMChain(llm=llm, prompt=prompt_evaluacion)

# Prompt para generar una pregunta de seguimiento sobre la noticia
prompt_pregunta = PromptTemplate(
    template="""
    Respuesta del inversor: {respuesta}
    
    La respuesta no aborda de manera suficiente los aspectos de la noticia relacionados con ESG (Ambiental, Social y Gobernanza) o con el riesgo. 
    Por favor, ¿podrías profundizar en qué aspectos específicos de la noticia te preocupan, ya sea en términos de sostenibilidad o de riesgo?
    """,
    input_variables=["respuesta"]
)
cadena_pregunta = LLMChain(llm=llm, prompt=prompt_pregunta)

# Prompt para generar el perfil del inversor
prompt_perfil = PromptTemplate(
    template="""
    Análisis de reacciones: {analisis}
    
    Genera un perfil detallado del inversor basado en sus reacciones, enfocándote en ESG (Ambiental, Social y Gobernanza) y su aversión al riesgo. 
    Asigna una puntuación de 0 a 100 para cada pilar ESG y para el riesgo, en este formato:
    Ambiental: [puntuación], Social: [puntuación], Gobernanza: [puntuación], Riesgo: [puntuación]
    """,
    input_variables=["analisis"]
)
cadena_perfil = LLMChain(llm=llm, prompt=prompt_perfil)

# Inicializar el estado de la sesión
if "historial" not in st.session_state:
    st.session_state.historial = []
    st.session_state.contador = 0
    st.session_state.reacciones = []
    st.session_state.mostrada_noticia = False

# Interfaz de usuario
st.title("Chatbot de Análisis de Sentimiento")

# Mostrar historial de conversación
for mensaje in st.session_state.historial:
    with st.chat_message(mensaje["tipo"]):
        st.write(mensaje["contenido"])

# Mostrar noticias y recoger respuestas del usuario
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

        # Evaluar si la respuesta es suficiente
        evaluacion = cadena_evaluacion.run(respuesta=user_input)
        suficiente = re.search(r"Suficiente: (Sí|No)", evaluacion)

        if suficiente and suficiente.group(1) == "No":
            # Generar una pregunta de seguimiento centrada en los aspectos de la noticia (ESG o riesgo)
            pregunta_followup = cadena_pregunta.run(respuesta=user_input)
            
            with st.chat_message("bot", avatar="🤖"):
                st.write(f"{pregunta_followup}")
            
            st.session_state.historial.append({"tipo": "bot", "contenido": pregunta_followup})
        
        else:
            # Pasar a la siguiente noticia si la respuesta es suficiente
            st.session_state.contador += 1
            st.session_state.mostrada_noticia = False
            st.rerun()

else:
    # Generar el perfil final del inversor
    analisis_total = "\n".join(st.session_state.reacciones)
    perfil = cadena_perfil.run(analisis=analisis_total)
    
    with st.chat_message("bot", avatar="🤖"):
        st.write(f"**Perfil del inversor:** {perfil}")
    
    st.session_state.historial.append({"tipo": "bot", "contenido": f"**Perfil del inversor:** {perfil}"})

    # Extraer puntuaciones con regex
    puntuaciones = {
        "Ambiental": int(re.search(r"Ambiental: (\d+)", perfil).group(1)),
        "Social": int(re.search(r"Social: (\d+)", perfil).group(1)),
        "Gobernanza": int(re.search(r"Gobernanza: (\d+)", perfil).group(1)),
        "Riesgo": int(re.search(r"Riesgo: (\d+)", perfil).group(1)),
    }

    # Crear gráfico de perfil del inversor
    categorias = list(puntuaciones.keys())
    valores = list(puntuaciones.values())

    fig, ax = plt.subplots()
    ax.bar(categorias, valores)
    ax.set_ylabel("Puntuación (0-100)")
    ax.set_title("Perfil del Inversor")
    st.pyplot(fig)

    # Guardar en Google Sheets
    try:
        creds_json_str = st.secrets["gcp_service_account"]
        creds_json = json.loads(creds_json_str)
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_json, scope)
        client = gspread.authorize(creds)
        sheet = client.open('BBDD_RESPUESTAS').sheet1

        fila = st.session_state.reacciones[:] + [
            puntuaciones["Ambiental"],
            puntuaciones["Social"],
            puntuaciones["Gobernanza"],
            puntuaciones["Riesgo"]
        ]
        
        sheet.append_row(fila)
        st.success("Respuestas y perfil guardados en Google Sheets.")
    except Exception as e:
        st.error(f"Error al guardar datos: {e}")
    except Exception as e:
        st.error(f"Error al guardar datos: {e}")
