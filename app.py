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

# Plantilla para evaluación de suficiencia
plantilla_evaluacion = """
Evalúa si esta respuesta del usuario es suficientemente detallada para un análisis ESG. 
Considera como criterios:
- Claridad de la opinión expresada
- Especificidad respecto a la noticia
- Mención de aspectos relevantes (ambiental, social, gobernanza o riesgo)
- Expresión de preocupaciones o riesgos identificables

Respuesta del usuario: {respuesta}

Si la respuesta es vaga, demasiado breve o no menciona aspectos concretos, devuelve "False".
Si contiene una opinión sustancial con elementos analizables, devuelve "True".

Solo devuelve "True" o "False".
"""
prompt_evaluacion = PromptTemplate(template=plantilla_evaluacion, input_variables=["respuesta"])
cadena_evaluacion = LLMChain(llm=llm, prompt=prompt_evaluacion)

# Plantilla simplificada para reacciones
plantilla_reaccion = """
Reacción del inversor: {reaccion}
Analiza el sentimiento y la preocupación expresada.  
Clasifica la preocupación principal en una de estas categorías:  
- Ambiental  
- Social  
- Gobernanza  
- Riesgo  

Genera ÚNICAMENTE una pregunta de seguimiento enfocada en la categoría detectada para profundizar en la opinión del inversor, SIN incluir análisis ni explicaciones.  
Ejemplo de formato:  
"¿Consideras que la existencia de mecanismos robustos de control interno y transparencia podría mitigar tu preocupación por la gobernanza corporativa en esta empresa?"
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

# Función para procesar respuestas válidas
def procesar_respuesta_valida(user_input):
    # Contador de preguntas por noticia
    if "contador_preguntas" not in st.session_state:
        st.session_state.contador_preguntas = 0
    
    pregunta_seguimiento = cadena_reaccion.run(reaccion=user_input).strip()
    
    with st.chat_message("bot", avatar="🤖"):
        if "¿" in pregunta_seguimiento and st.session_state.contador_preguntas < 2:
            st.write(f"**Pregunta de seguimiento:** {pregunta_seguimiento}")
            st.session_state.pregunta_pendiente = pregunta_seguimiento
            st.session_state.contador_preguntas += 1
        else:
            st.write("Gracias por tus respuestas. Avanzando a la siguiente noticia...")
            st.session_state.contador += 1
            st.session_state.mostrada_noticia = False
            st.session_state.contador_preguntas = 0
    
    st.session_state.historial.append({"tipo": "bot", "contenido": pregunta_seguimiento})
    st.session_state.reacciones.append(user_input)
    st.session_state.esperando_ampliacion = False
    st.rerun()

# Inicialización de estados
if "historial" not in st.session_state:
    st.session_state.historial = []
    st.session_state.contador = 0
    st.session_state.reacciones = []
    st.session_state.mostrada_noticia = False
    st.session_state.esperando_ampliacion = False
    st.session_state.contador_preguntas = 0

st.title("Chatbot de Análisis de Sentimiento")

# Mostrar historial de chat
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
        st.session_state.esperando_ampliacion = False
        if hasattr(st.session_state, 'pregunta_pendiente'):
            del st.session_state.pregunta_pendiente

    user_input = st.chat_input("Escribe tu respuesta aquí...")
    if user_input:
        st.session_state.historial.append({"tipo": "user", "contenido": user_input})
        
        if hasattr(st.session_state, 'pregunta_pendiente'):
            del st.session_state.pregunta_pendiente
            procesar_respuesta_valida(user_input)
        else:
            if not st.session_state.esperando_ampliacion:
                evaluacion = cadena_evaluacion.run(respuesta=user_input).strip().lower()
                
                if evaluacion == "false":
                    st.session_state.esperando_ampliacion = True
                    with st.chat_message("bot", avatar="🤖"):
                        st.markdown(f"¿Podrías ampliar tu opinión sobre: **{noticias[st.session_state.contador]}**?")
                    st.session_state.historial.append({"tipo": "bot", "contenido": "Solicitud de ampliación"})
                else:
                    procesar_respuesta_valida(user_input)
            else:
                procesar_respuesta_valida(user_input)
else:
    # Generación del perfil final
    analisis_total = "\n".join(st.session_state.reacciones)
    perfil = cadena_perfil.run(analisis=analisis_total)
    with st.chat_message("bot", avatar="🤖"):
        st.write(f"**Perfil del inversor:** {perfil}")
    st.session_state.historial.append({"tipo": "bot", "contenido": f"**Perfil del inversor:** {perfil}"})

    puntuaciones = {
        "Ambiental": int(re.search(r"Ambiental: (\d+)", perfil).group(1)),
        "Social": int(re.search(r"Social: (\d+)", perfil).group(1)),
        "Gobernanza": int(re.search(r"Gobernanza: (\d+)", perfil).group(1)),
        "Riesgo": int(re.search(r"Riesgo: (\d+)", perfil).group(1))
