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

# Prompt para evaluar si la respuesta es suficientemente detallada
plantilla_evaluacion = """
Respuesta del inversor: {respuesta}
Evalúa si la respuesta es suficientemente detallada o si es vaga e inespecífica.
Una respuesta detallada debe incluir una emoción clara, una preocupación específica o una evaluación del impacto.

Si la respuesta es demasiado genérica o ambigua, devuelve:
"Suficiente: No"
Si la respuesta es clara y bien fundamentada, devuelve:
"Suficiente: Sí"
"""
prompt_evaluacion = PromptTemplate(template=plantilla_evaluacion, input_variables=["respuesta"])
cadena_evaluacion = LLMChain(llm=llm, prompt=prompt_evaluacion)

# Prompt para analizar la reacción del usuario y generar una pregunta de seguimiento
plantilla_reaccion = """
Reacción del inversor: {reaccion}
Analiza el sentimiento y la preocupación expresada.  
Clasifica la preocupación principal en una de estas categorías:  
- Ambiental  
- Social  
- Gobernanza  
- Riesgo  

Si la respuesta es demasiado breve o poco clara, solicita más detalles de manera específica.  

Luego, genera una pregunta de seguimiento enfocada en la categoría detectada para profundizar en la opinión del inversor.  
Por ejemplo:  
- Si la categoría es Ambiental: "¿Cómo crees que esto afecta la sostenibilidad del sector?"  
- Si la categoría es Social: "¿Crees que esto puede afectar la percepción pública de la empresa?"  
- Si la categoría es Gobernanza: "¿Este evento te hace confiar más o menos en la gestión de la empresa?"  
- Si la categoría es Riesgo: "¿Consideras que esto aumenta la incertidumbre en el mercado?"  

Devuelve la categoría detectada y la pregunta de seguimiento en el siguiente formato:  
Categoría: [nombre de la categoría]  
Pregunta de seguimiento: [pregunta generada]
"""
prompt_reaccion = PromptTemplate(template=plantilla_reaccion, input_variables=["reaccion"])
cadena_reaccion = LLMChain(llm=llm, prompt=prompt_reaccion)

# Prompt para generar perfil ESG y de riesgo
plantilla_perfil = """
Análisis de reacciones: {analisis}
Genera un perfil detallado del inversor basado en sus reacciones, enfocándote en los pilares ESG (Ambiental, Social y Gobernanza) y su aversión al riesgo. 
Asigna una puntuación de 0 a 100 para cada pilar ESG y para el riesgo, donde 0 indica ninguna preocupación y 100 máxima preocupación o aversión.
Devuelve las 4 puntuaciones en formato: Ambiental: [puntuación], Social: [puntuación], Gobernanza: [puntuación], Riesgo: [puntuación]
"""
prompt_perfil = PromptTemplate(template=plantilla_perfil, input_variables=["analisis"])
cadena_perfil = LLMChain(llm=llm, prompt=prompt_perfil)

# Inicializar estado en Streamlit
if "historial" not in st.session_state:
    st.session_state.historial = []
    st.session_state.contador = 0
    st.session_state.reacciones = []
    st.session_state.mostrada_noticia = False

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

        # Evaluamos si la respuesta es suficientemente detallada
        evaluacion = cadena_evaluacion.run(respuesta=user_input)
        suficiente_match = re.search(r"Suficiente: (Sí|No)", evaluacion)

        if suficiente_match and suficiente_match.group(1) == "No":
            analisis_reaccion = cadena_reaccion.run(reaccion=user_input)
            pregunta_match = re.search(r"Pregunta de seguimiento: (.+)", analisis_reaccion)

            with st.chat_message("bot", avatar="🤖"):
                if pregunta_match:
                    st.write(f"{pregunta_match.group(1)}")
                else:
                    st.write("¿Podrías dar más detalles sobre tu opinión?")

            st.session_state.historial.append({"tipo": "bot", "contenido": pregunta_match.group(1) if pregunta_match else "¿Podrías dar más detalles sobre tu opinión?"})
        else:
            st.session_state.contador += 1
            st.session_state.mostrada_noticia = False
            st.rerun()
else:
    analisis_total = "\n".join(st.session_state.reacciones)
    perfil = cadena_perfil.run(analisis=analisis_total)
    with st.chat_message("bot", avatar="🤖"):
        st.write(f"**Perfil del inversor:** {perfil}")

    st.session_state.historial.append({"tipo": "bot", "contenido": f"**Perfil del inversor:** {perfil}"})

    # Guardar en Google Sheets (opcional)
    # Aquí podrías agregar la lógica para almacenar el perfil en la base de datos

st.success("Respuestas y perfil guardados correctamente.")  
