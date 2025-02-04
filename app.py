import streamlit as st
import requests
import io
from PIL import Image

API_URL = "http://localhost:8000/generate"  # Адрес FastAPI-сервера

st.title("Смена стиля изображений")
st.write("Загрузите фотографию для смены стиля")


content_image = st.file_uploader("Выберите изображение контента", type=["jpg", "png"])

model_choice = st.selectbox("Выберите модель", ["CycleGAN", "Stable Diffusion"])
style_strength = st.slider("Уровень стилизации", 0.0, 1.0, 0.5)  # Добавлен слайдер

if st.button("Применить стиль"):
    if  content_image:
        files = {
            "content": (content_image.name, content_image.getvalue(), content_image.type)
        }
        data = {
            "model": model_choice,
            "style_strength": style_strength
        }
        response = requests.post(API_URL, files=files, data=data)
        print("Отправляемые данные:", data, content_image.type)
        if response.status_code == 200:
            output_image = Image.open(io.BytesIO(response.content))
            st.image(output_image, caption="Результат переноса стиля", use_container_width=True)
            st.image(content_image , caption="Оригинал", use_container_width=True)
            save_path = "saved_image1.png"
            output_image.save(save_path)
            st.write(f"Изображение сохранено как {save_path}")
        else:а названия классов в метках недолжно быть?
            st.error("Ошибка обработки изображения")
