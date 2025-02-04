import streamlit as st
import io
from PIL import Image
import torch
from torchvision import transforms
from diffusers import StableDiffusionPipeline
from cycle_gan import Cycle_gan  # Подключение CycleGAN

# Определяем устройство (CPU или GPU)
#device = "cuda" if torch.cuda.is_available() else "cpu" на стримлит лишнее

# Загружаем модели
cycle_gan = Cycle_gan()
stable_diffusion = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
#stable_diffusion.to(device)

transform = transforms.Compose([
    transforms.Resize(512, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(512),
])

# Интерфейс Streamlit
st.title("Смена стиля изображений")
st.write("Загрузите изображение для смены стиля")

content_image = st.file_uploader("Выберите изображение", type=["jpg", "png"])
model_choice = st.selectbox("Выберите модель", ["CycleGAN", "Stable Diffusion"])
style_strength = st.slider("Уровень стилизации", 0.0, 1.0, 0.5)

if st.button("Применить стиль"):
    if content_image:
        content_img = Image.open(content_image).convert("RGB")

        if model_choice == "CycleGAN":
            result_img = cycle_gan.transfer_style(content_img)  # Применение CycleGAN
            result_img = result_img.convert("RGB")
            content_img = transform(content_img)
            result_img = Image.blend(content_img, result_img, style_strength)  # Регулировка стиля
        else:
            prompt = "Lets style uploaded image like Monet"
            result_img = stable_diffusion(prompt=prompt, guidance_scale=style_strength * 10).images[0]

        original_size = content_img.size
        result_img = result_img.resize(original_size)

        st.image(result_img, caption="Результат переноса стиля", use_column_width=True)
        st.image(content_img, caption="Оригинал", use_column_width=True)
