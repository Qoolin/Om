from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import Response
import io
from PIL import Image
import torch
from torchvision import transforms
from diffusers import StableDiffusionPipeline
from cycle_gan import Cycle_gan  # Подключение CycleGAN

# Создаем FastAPI-приложение
app = FastAPI()

# Определяем устройство (CPU или GPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Загружаем модели
cycle_gan = Cycle_gan()
stable_diffusion = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
stable_diffusion.to(device)

transform = transforms.Compose([
    transforms.Resize(512, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(512),
    ])


@app.post("/generate")
async def generate_image(
    content: UploadFile = File(...),
    model: str = Form(...),
    style_strength: float = Form(...)
):
    # Открываем изображение
    content_img = Image.open(io.BytesIO(await content.read())).convert("RGB")

    if model == "CycleGAN":
        result_img = cycle_gan.transfer_style(content_img)# Применение CycleGAN
        result_img = result_img.convert("RGB")
        # Преобразования, чтобы исходное изображение совпадало с итоговым
        content_img = transform(content_img)
        result_img = Image.blend(content_img, result_img, style_strength)  # Регулировка стиля
    else:
        prompt = "style uploaded image like Claude Monet "
       # result_img = content_img.copy()
        result_img = stable_diffusion(prompt=prompt, guidance_scale=style_strength * 10).images[0]  # Stable Diffusion


    original_size = content_img.size
    # Сохраняем результат в тот же формат
    img_byte_arr = io.BytesIO()
    result_img = result_img.resize(original_size)
    result_img.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)
    # Возвращаем изображение как ответ с соответствующим MIME-типом
    return Response(content=img_byte_arr.getvalue(), media_type="image/png")

# Запуск сервера (если файл запускается напрямую)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
