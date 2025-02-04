import torch
from PIL import Image
from Cycle_gan_train import Generator  # Импортируем класс модели
from pathlib import Path
from torchvision import transforms
import gdown
#Ссылка на веса
url = "https://drive.google.com/uc?id=1-O2wcB9VOKCUgPWEyMu1-n__RGO53m0y"
output = "cycle_gan.pth"  # имя файла весов
gdown.download(url, output, quiet=False)

class Cycle_gan:
    def __init__(self, model_path=None):
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Универсальный путь к модели в корневой папке проекта
        if model_path is None:
            model_path = Path(__file__).resolve().parent / "cycle_gan.pth"

        self.model = self.load_model(model_path)
        self.transform = transforms.Compose([
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Нормализация для RGB
        ])

    def load_model(self, model_path):
        """Загружает модель и переводит в режим инференса."""
        # Создаём генератор (3,3 - RGB в RGB)
        gen_AB = Generator(3, 3) #.to(self.device)

        # Загружаем сохранённые веса
        checkpoint = torch.load(model_path, map_location=self.device)
        gen_AB.load_state_dict(checkpoint['gen_AB'])
        gen_AB.eval()  # Переводим в режим инференса
        return gen_AB

    def transfer_style(self, content_img: Image):
        """Процесс стилизации изображения с использованием модели."""
        # Проверка и конвертация формата изображения
        if content_img.mode != "RGB":
            content_img = content_img.convert("RGB")

        # Преобразуем изображение для подачи в модель
        content_tensor = self.transform(content_img).unsqueeze(0).to(self.device)

        with torch.no_grad():  # Отключаем градиенты для инференса
            styled_img = self.model(content_tensor)

        # Денормализация и преобразование в диапазон [0, 255]
        styled_img = styled_img.squeeze(0).cpu()  # Перемещаем на CPU
        styled_img = styled_img * 0.5 + 0.5  # Денормализация
        styled_img = styled_img.clamp(0, 1)  # Ограничение значений

        # Преобразуем в изображение PIL
        styled_img_pil = transforms.ToPILImage()(styled_img.float())
        return styled_img_pil
