import os
import random
import unicodedata
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D

st.set_page_config(page_title="Joystick de mano", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bungee&display=swap');
.stApp {
    background-color: #14081f;
}
[data-testid="stHeader"] {
    background: transparent;
}
section[data-testid="stSidebar"] {
    background-color: #1f0f33;
}
.stApp p, .stApp label, .stApp li {
    color: #f5e9ff;
}
h1, h2, h3 {
    font-family: 'Bungee', sans-serif;
    color: #00e5ff !important;
}
[data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
    color: #f5e9ff !important;
}
[data-testid="stAlert"] {
    background-color: #2a1347 !important;
    border: 1px solid #ff2e97;
}
[data-testid="stAlert"] * {
    color: #f5e9ff !important;
}
[data-testid="stFileUploaderDropzone"] {
    background-color: #1f0f33 !important;
    border: 1px dashed #ff2e97;
}
[data-testid="stFileUploaderDropzone"] * {
    color: #f5e9ff !important;
}
[data-testid="stFileUploaderDropzone"] button {
    background-color: #ff2e97 !important;
    color: #ffffff !important;
}
div.stButton > button {
    background-color: #ff2e97;
    color: #ffffff !important;
    border: none;
    border-radius: 8px;
    font-weight: bold;
    padding: 8px 24px;
}
div.stButton > button:hover {
    background-color: #00e5ff;
    color: #14081f !important;
}
</style>
""", unsafe_allow_html=True)

TAM = 6
movimientos = {"izquierda": (-1, 0), "derecha": (1, 0), "arriba": (0, -1), "abajo": (0, 1)}
flechas = {"izquierda": "⬅️", "derecha": "➡️", "arriba": "⬆️", "abajo": "⬇️"}


class DepthwiseCompat(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop("groups", None)
        super().__init__(*args, **kwargs)


@st.cache_resource
def cargar():
    modelo = load_model("keras_model.h5", compile=False, custom_objects={"DepthwiseConv2D": DepthwiseCompat})
    with open("labels.txt", encoding="utf-8") as f:
        etiquetas = [" ".join(linea.strip().split(" ")[1:]) for linea in f if linea.strip()]
    return modelo, etiquetas


def predecir(modelo, imagen):
    img = ImageOps.fit(imagen, (224, 224), Image.Resampling.LANCZOS)
    datos = (np.asarray(img).astype(np.float32) / 127.5) - 1
    return modelo.predict(datos[np.newaxis, ...], verbose=0)[0]


def limpiar(texto):
    sin_tildes = "".join(c for c in unicodedata.normalize("NFD", texto.lower().strip()) if unicodedata.category(c) != "Mn")
    return sin_tildes


def nueva_meta(pos):
    while True:
        meta = [random.randrange(TAM), random.randrange(TAM)]
        if meta != pos:
            return meta


def reiniciar():
    st.session_state.pos = [0, 0]
    st.session_state.meta = nueva_meta([0, 0])
    st.session_state.pasos = 0
    st.session_state.llegadas = 0
    st.session_state.pop("ronda", None)


def dibujar(pos, meta):
    filas = []
    for y in range(TAM):
        fila = ""
        for x in range(TAM):
            if [x, y] == pos:
                fila += "🤖"
            elif [x, y] == meta:
                fila += "⭐"
            else:
                fila += "⬛"
        filas.append(fila)
    st.markdown('<div style="font-size:34px; line-height:1.25; text-align:center">' + "<br>".join(filas) + "</div>", unsafe_allow_html=True)


if "pos" not in st.session_state:
    reiniciar()

with st.sidebar:
    st.subheader("Reglas del juego")
    st.write("1. Señala una dirección con la mano.")
    st.write("2. Toma la foto.")
    st.write("3. La IA reconoce el gesto y mueve el robot una casilla.")
    st.write("Llega a la ⭐ en el menor número de pasos.")
    minimo = st.slider("Confianza mínima", 0.3, 0.95, 0.6, 0.05)
    invertir = st.checkbox("Invertir izquierda y derecha")
    st.button("Reiniciar juego", on_click=reiniciar)

st.title("🕹️ Joystick de mano")
st.write("Tu mano es el control. Guía al robot hasta la estrella con gestos.")

if not (os.path.exists("keras_model.h5") and os.path.exists("labels.txt")):
    st.error("Faltan los archivos keras_model.h5 y labels.txt. Súbelos a la misma carpeta que app.py.")
    st.stop()

try:
    modelo, etiquetas = cargar()
except Exception as e:
    st.error(f"No se pudo cargar el modelo: {e}")
    st.stop()

tablero = st.container()

modo = st.radio("¿De dónde viene la foto?", ["Tomar foto", "Subir imagen"], horizontal=True)

if modo == "Tomar foto":
    foto = st.camera_input("Señala una dirección y toma la foto")
else:
    foto = st.file_uploader("Sube una foto de tu gesto", type=["jpg", "jpeg", "png"])

if foto is not None:
    imagen = ImageOps.exif_transpose(Image.open(foto)).convert("RGB")
    huella = hash(foto.getvalue())
    if st.session_state.get("ultima") != huella:
        st.session_state.ultima = huella
        probs = predecir(modelo, imagen)
        indice = int(np.argmax(probs))
        gesto = limpiar(etiquetas[indice])
        if invertir and gesto in ("izquierda", "derecha"):
            gesto = "derecha" if gesto == "izquierda" else "izquierda"
        ronda = {"probs": probs, "indice": indice, "gesto": gesto, "movio": False, "llego": False}
        if gesto in movimientos and probs[indice] >= minimo:
            dx, dy = movimientos[gesto]
            x = min(max(st.session_state.pos[0] + dx, 0), TAM - 1)
            y = min(max(st.session_state.pos[1] + dy, 0), TAM - 1)
            st.session_state.pos = [x, y]
            st.session_state.pasos += 1
            ronda["movio"] = True
            if [x, y] == st.session_state.meta:
                st.session_state.llegadas += 1
                st.session_state.meta = nueva_meta([x, y])
                ronda["llego"] = True
        st.session_state.ronda = ronda

with tablero:
    st.header("Tablero")
    dibujar(st.session_state.pos, st.session_state.meta)

ronda = st.session_state.get("ronda")

if ronda and foto is not None:
    probs = ronda["probs"]
    indice = ronda["indice"]
    st.divider()
    st.write(f"La IA ve: **{etiquetas[indice]}** {flechas.get(ronda['gesto'], '')} ({probs[indice]:.0%})")
    if ronda["llego"]:
        st.success("⭐ ¡Llegaste a la estrella! Aparece una nueva.")
    elif ronda["movio"]:
        st.info("🤖 Movimiento hecho.")
    elif ronda["gesto"] in movimientos:
        st.warning("No estoy seguro de tu gesto. Acércate un poco, con buena luz, y vuelve a intentarlo.")
    else:
        st.info("No veo un gesto de dirección.")
    st.bar_chart(pd.Series(probs, index=etiquetas))

st.divider()
m1, m2 = st.columns(2)
m1.metric("Pasos", st.session_state.pasos)
m2.metric("Estrellas", st.session_state.llegadas)
