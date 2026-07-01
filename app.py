import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="Detector por Regras", layout="centered")

st.title("🔍 Sistema Simples de Detecção (Regras + OpenCV)")

uploaded_file = st.file_uploader("Envie uma imagem", type=["jpg", "png", "jpeg"])

# -------------------------
# FUNÇÃO PRINCIPAL
# -------------------------

def analyze_image(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 70, 150)

    h, w = edges.shape

    # ----------------------
    # SCORE PESSOA
    # ----------------------
    person_region = edges[0:int(h * 0.6), int(w * 0.3):int(w * 0.7)]
    person_score = np.mean(person_region)

    # ----------------------
    # SCORE CARRO / ANIMAL
    # ----------------------
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    car_score = 0
    animal_score = 0

    for c in contours:

        x, y, cw, ch = cv2.boundingRect(c)

        area = cw * ch
        aspect = cw / (ch + 1)

        # CARRO (formas grandes e horizontais)
        if area > 12000 and aspect > 1.8:
            car_score += 1

        # ANIMAL (formas irregulares)
        peri = cv2.arcLength(c, True)

        if peri > 0:
            circularity = 4 * np.pi * cv2.contourArea(c) / (peri * peri)

            if circularity < 0.25:
                animal_score += 1

    # ----------------------
    # DECISÃO FINAL
    # ----------------------

    person = person_score > 40
    car = car_score > 0
    animal = animal_score > 3

    scores = {
        "Pessoa": person_score,
        "Carro": car_score,
        "Animal": animal_score
    }

    winner = max(scores, key=scores.get)

    return person, car, animal, winner


# -------------------------
# STREAMLIT APP
# -------------------------

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        st.error("Erro ao carregar a imagem.")
        st.stop()

    st.image(img, channels="BGR", caption="Imagem enviada")

    person, car, animal, winner = analyze_image(img)

    st.subheader("📊 Resultado (Regras Heurísticas)")

    st.write("👤 Pessoa:", "SIM" if person else "NÃO")
    st.write("🚗 Carro:", "SIM" if car else "NÃO")
    st.write("🐾 Animal:", "SIM" if animal else "NÃO")

    st.subheader("🏆 Classe dominante (simulação)")
    st.write(winner)
