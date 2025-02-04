import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

torch.manual_seed(42)


def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    '''
    Функция для визуализации изображений: для заданного тензора изображений,
    количества изображений и размера каждого изображения, функция строит и
    отображает изображения в виде сетки.
    '''
    image_tensor = (image_tensor + 1) / 2
    image_shifted = image_tensor
    image_unflat = image_shifted.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()


import glob
import random
import os
from torch.utils.data import Dataset
from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, root, transform=None, mode='train'):
        self.transform = transform
        self.files_A = sorted(glob.glob(os.path.join(root, '%sA' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%sB' % mode) + '/*.*'))
        if len(self.files_A) > len(self.files_B):
            self.files_A, self.files_B = self.files_B, self.files_A
        self.new_perm()
      #  assert len(self.files_A) > 0, "Make sure you downloaded the images!"

    def new_perm(self):
        self.randperm = torch.randperm(len(self.files_B))[:len(self.files_A)]

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))
        item_B = self.transform(Image.open(self.files_B[self.randperm[index]]))
        if item_A.shape[0] != 3:
            item_A = item_A.repeat(3, 1, 1)
        if item_B.shape[0] != 3:
            item_B = item_B.repeat(3, 1, 1)
        if index == len(self) - 1:
            self.new_perm()
        # Old versions of PyTorch didn't support normalization for different-channeled images
        return (item_A - 0.5) * 2, (item_B - 0.5) * 2

    def __len__(self):
        return min(len(self.files_A), len(self.files_B))


class ResidualBlock(nn.Module):
    '''
     Класс ResidualBlock (Остаточный блок): Выполняет две свертки и
     нормализацию экземпляра, входные данные добавляются к этому выходу,
     чтобы сформировать выход остаточного блока.
     Значения: input_channels: количество каналов, ожидаемых от заданного входа
    '''

    def __init__(self, input_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, padding_mode='reflect')
        self.conv2 = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, padding_mode='reflect')
        self.instancenorm = nn.InstanceNorm2d(input_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        '''
        Функция для выполнения прямого прохода ResidualBlock: Для заданного
        тензора изображения, функция выполняет операции остаточного блока и
        возвращает преобразованный тензор. Параметры: x: тензор изображения
        формы (размер пакета, количество каналов, высота, ширина)
        '''
        original_x = x.clone()
        x = self.conv1(x)
        x = self.instancenorm(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.instancenorm(x)
        return original_x + x


class ContractingBlock(nn.Module):
    '''
    Класс ContractingBlock (Сжимающий блок): Выполняет свертку, за которой
    следует операция максимального пулинга и опциональная нормализация
    экземпляра. Значения: input_channels: количество каналов, ожидаемых от
    заданного входа
    '''

    def __init__(self, input_channels, use_bn=True, kernel_size=3, activation='relu'):
        super(ContractingBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, input_channels * 2, kernel_size=kernel_size, padding=1, stride=2,
                               padding_mode='reflect')
        self.activation = nn.ReLU() if activation == 'relu' else nn.LeakyReLU(0.2)
        if use_bn:
            self.instancenorm = nn.InstanceNorm2d(input_channels * 2)
        self.use_bn = use_bn

    def forward(self, x):
        '''
        Функция для выполнения прямого прохода ContractingBlock:
        Для заданного тензора изображения, функция выполняет операции
        сжимающего блока и возвращает преобразованный тензор.
        Параметры: x: тензор изображения формы
        (размер пакета, количество каналов, высота, ширина)
        '''
        x = self.conv1(x)
        if self.use_bn:
            x = self.instancenorm(x)
        x = self.activation(x)
        return x


class ExpandingBlock(nn.Module):
    '''
    Класс ExpandingBlock (Расширяющий блок): Выполняет операцию
    транспонированной свертки для увеличения разрешения, с опциональной
    нормализацией экземпляра. Значения: input_channels: количество каналов,
    ожидаемых от заданного входа
    '''

    def __init__(self, input_channels, use_bn=True):
        super(ExpandingBlock, self).__init__()
        self.conv1 = nn.ConvTranspose2d(input_channels, input_channels // 2, kernel_size=3, stride=2, padding=1,
                                        output_padding=1)
        if use_bn:
            self.instancenorm = nn.InstanceNorm2d(input_channels // 2)
        self.use_bn = use_bn
        self.activation = nn.ReLU()

    def forward(self, x):
        '''
        Функция для выполнения прямого прохода ExpandingBlock:
        Для заданного тензора изображения, функция выполняет операции
        расширяющего блока и возвращает преобразованный тензор.
        Параметры: x: тензор изображения формы (размер пакета,
        количество каналов, высота, ширина) skip_con_x:
        тензор изображения из сжимающего пути (от противолежащего блока x)
        для пропуска соединения
        '''
        x = self.conv1(x)
        if self.use_bn:
            x = self.instancenorm(x)
        x = self.activation(x)
        return x


class FeatureMapBlock(nn.Module):
    '''
    Класс FeatureMapBlock: Финальный слой Генератора - преобразует
    каждый выход в желаемое количество выходных каналов
    Значения: input_channels: количество каналов, ожидаемых от заданного входа
    output_channels: количество каналов, ожидаемых для заданного выхода
    '''

    def __init__(self, input_channels, output_channels):
        super(FeatureMapBlock, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=7, padding=3, padding_mode='reflect')

    def forward(self, x):
        '''
        Функция для выполнения прямого прохода FeatureMapBlock:
        Для заданного тензора изображения, функция возвращает его,
        преобразованным в желаемое количество каналов.
        Параметры: x: тензор изображения формы
        (размер пакета, количество каналов, высота, ширина)
        '''
        x = self.conv(x)
        return x


class Generator(nn.Module):
    '''
    Класс Генератора
    Состоит из 2 сверточных (сжимающих) блоков, 9 остаточных (residual) блоков
    и 2 разворачивающих (расширяющих) блоков. Преобразует входное изображение в
    изображение из другого класса. Также включает слой upfeature в начале и
    слой downfeature в конце.

    Аргументы:
        input_channels: количество каналов во входном изображении
        output_channels: количество каналов на выходе
    '''

    def __init__(self, input_channels, output_channels, hidden_channels=64):
        super(Generator, self).__init__()
        self.upfeature = FeatureMapBlock(input_channels, hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels)
        self.contract2 = ContractingBlock(hidden_channels * 2)
        res_mult = 4
        self.res0 = ResidualBlock(hidden_channels * res_mult)
        self.res1 = ResidualBlock(hidden_channels * res_mult)
        self.res2 = ResidualBlock(hidden_channels * res_mult)
        self.res3 = ResidualBlock(hidden_channels * res_mult)
        self.res4 = ResidualBlock(hidden_channels * res_mult)
        self.res5 = ResidualBlock(hidden_channels * res_mult)
        self.res6 = ResidualBlock(hidden_channels * res_mult)
        self.res7 = ResidualBlock(hidden_channels * res_mult)
        self.res8 = ResidualBlock(hidden_channels * res_mult)
        self.expand2 = ExpandingBlock(hidden_channels * 4)
        self.expand3 = ExpandingBlock(hidden_channels * 2)
        self.downfeature = FeatureMapBlock(hidden_channels, output_channels)
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        '''
        Функция для выполнения прямого прохода через генератор:
        принимает на вход изображение, пропускает его через U-Net с
        остаточными блоками и возвращает результат.

        Аргументы:
            x: тензор изображения размером (batch size, channels, height, width)
        '''
        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.res0(x2)
        x4 = self.res1(x3)
        x5 = self.res2(x4)
        x6 = self.res3(x5)
        x7 = self.res4(x6)
        x8 = self.res5(x7)
        x9 = self.res6(x8)
        x10 = self.res7(x9)
        x11 = self.res8(x10)
        x12 = self.expand2(x11)
        x13 = self.expand3(x12)
        xn = self.downfeature(x13)
        return self.tanh(xn)


class Discriminator(nn.Module):
    '''
    Класс Дискриминатора
    Имеет структуру, похожую на сжимающую (contracting) часть U-Net.
    Дискриминатор принимает изображение и выдает матрицу значений,
    определяя, какие области изображения являются реальными, а какие - фейковыми.

    Аргументы:
        input_channels: количество каналов во входном изображении
        hidden_channels: начальное количество фильтров в сверточных слоях дискриминатора
    '''

    def __init__(self, input_channels, hidden_channels=64):
        super(Discriminator, self).__init__()
        self.upfeature = FeatureMapBlock(input_channels, hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels, use_bn=False, kernel_size=4, activation='lrelu')
        self.contract2 = ContractingBlock(hidden_channels * 2, kernel_size=4, activation='lrelu')
        self.contract3 = ContractingBlock(hidden_channels * 4, kernel_size=4, activation='lrelu')
        self.final = nn.Conv2d(hidden_channels * 8, 1, kernel_size=1)

    def forward(self, x):
        '''
        Функция для выполнения прямого прохода через дискриминатор.

        Аргументы:
            x: входное изображение

        Возвращает:
            xn: матрица вероятностей, указывающая, какие области изображения
                являются настоящими, а какие - сгенерированными.
        '''
        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        xn = self.final(x3)
        return xn


import torch.nn.functional as F

adv_criterion = nn.MSELoss()
recon_criterion = nn.L1Loss()

n_epochs = 100
dim_A = 3
dim_B = 3
display_step = 1000
batch_size = 1
lr = 0.0002
load_shape = 128
target_shape = 128
device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize(load_shape),
    transforms.RandomCrop(target_shape),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

import torchvision

dataset = ImageDataset("/kaggle/input/monet2photo", transform=transform)

gen_AB = Generator(dim_A, dim_B).to(device)
gen_BA = Generator(dim_B, dim_A).to(device)
gen_opt = torch.optim.Adam(list(gen_AB.parameters()) + list(gen_BA.parameters()), lr=lr, betas=(0.5, 0.999))
disc_A = Discriminator(dim_A).to(device)
disc_A_opt = torch.optim.Adam(disc_A.parameters(), lr=lr, betas=(0.5, 0.999))
disc_B = Discriminator(dim_B).to(device)
disc_B_opt = torch.optim.Adam(disc_B.parameters(), lr=lr, betas=(0.5, 0.999))


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)


# Feel free to change pretrained to False if you're training the model from scratch
pretrained = False
if pretrained:
    pre_dict = torch.load('/kaggle/input/cyclegan-monet2photo/pytorch/first-version/1/cycleGAN.pth')
    gen_AB.load_state_dict(pre_dict['gen_AB'])
    gen_BA.load_state_dict(pre_dict['gen_BA'])
    gen_opt.load_state_dict(pre_dict['gen_opt'])
    disc_A.load_state_dict(pre_dict['disc_A'])
    disc_A_opt.load_state_dict(pre_dict['disc_A_opt'])
    disc_B.load_state_dict(pre_dict['disc_B'])
    disc_B_opt.load_state_dict(pre_dict['disc_B_opt'])
    print("Loaded pre-trained model")
else:
    gen_AB = gen_AB.apply(weights_init)
    gen_BA = gen_BA.apply(weights_init)
    disc_A = disc_A.apply(weights_init)
    disc_B = disc_B.apply(weights_init)


def get_disc_loss(real_X, fake_X, disc_X, adv_criterion):
    '''
    Возвращает функцию потерь дискриминатора.
    Параметры:
        real_X: реальные изображения из набора X
        fake_X: сгенерированные изображения класса X
        disc_X: дискриминатор для класса X; принимает изображения и возвращает матрицы предсказаний (реальное/фейковое изображение)
        adv_criterion: функция потерь для состязательного обучения; принимает предсказания дискриминатора и целевые метки, возвращая
                       состязательную ошибку (её необходимо минимизировать)
    '''
    disc_fake_X_hat = disc_X(fake_X.detach())  # Отсоединяем от генератора
    disc_fake_X_loss = adv_criterion(disc_fake_X_hat, torch.zeros_like(disc_fake_X_hat))
    disc_real_X_hat = disc_X(real_X)
    disc_real_X_loss = adv_criterion(disc_real_X_hat, torch.ones_like(disc_real_X_hat))
    disc_loss = (disc_fake_X_loss + disc_real_X_loss) / 2
    return disc_loss


def get_gen_adversarial_loss(real_X, disc_Y, gen_XY, adv_criterion):
    '''
    Возвращает функцию состязательной потери генератора (и сгенерированные изображения для тестирования).
    Параметры:
        real_X: реальные изображения из набора X
        disc_Y: дискриминатор для класса Y; принимает изображения и возвращает матрицы предсказаний (реальное/фейковое изображение)
        gen_XY: генератор для преобразования изображений из класса X в класс Y
        adv_criterion: функция потерь для состязательного обучения; принимает предсказания дискриминатора и целевые метки, возвращая
                       состязательную ошибку (её необходимо минимизировать)
    '''
    fake_Y = gen_XY(real_X)
    disc_fake_Y_hat = disc_Y(fake_Y)
    adversarial_loss = adv_criterion(disc_fake_Y_hat, torch.ones_like(disc_fake_Y_hat))
    return adversarial_loss, fake_Y


def get_identity_loss(real_X, gen_YX, identity_criterion):
    '''
    Возвращает функцию потерь идентичности генератора (и сгенерированные изображения для тестирования).
    Параметры:
        real_X: реальные изображения из набора X
        gen_YX: генератор для преобразования изображений из класса Y в класс X
        identity_criterion: функция потерь идентичности; принимает реальные изображения X и изображения, прошедшие через генератор Y->X,
                            и возвращает ошибку идентичности (её необходимо минимизировать)
    '''
    identity_X = gen_YX(real_X)
    identity_loss = identity_criterion(identity_X, real_X)
    return identity_loss, identity_X


def get_cycle_consistency_loss(real_X, fake_Y, gen_YX, cycle_criterion):
    '''
    Возвращает функцию потерь циклической согласованности генератора (и сгенерированные изображения для тестирования).
    Параметры:
        real_X: реальные изображения из набора X
        fake_Y: сгенерированные изображения класса Y
        gen_YX: генератор для преобразования изображений из класса Y в класс X
        cycle_criterion: функция потерь циклической согласованности; принимает реальные изображения X, изображения,
                         прошедшие через генератор X->Y и затем Y->X, и возвращает ошибку циклической согласованности
                         (её необходимо минимизировать)
    '''
    cycle_X = gen_YX(fake_Y)
    cycle_loss = cycle_criterion(cycle_X, real_X)
    return cycle_loss, cycle_X


def get_gen_loss(real_A, real_B, gen_AB, gen_BA, disc_A, disc_B, adv_criterion, identity_criterion, cycle_criterion,
                 lambda_identity=0.1, lambda_cycle=10):
    '''
    Возвращает функцию потерь генератора.
    Параметры:
        real_A: реальные изображения из набора A
        real_B: реальные изображения из набора B
        gen_AB: генератор для преобразования изображений из класса A в класс B
        gen_BA: генератор для преобразования изображений из класса B в класс A
        disc_A: дискриминатор для класса A; принимает изображения и возвращает матрицы предсказаний (реальное/фейковое изображение)
        disc_B: дискриминатор для класса B; принимает изображения и возвращает матрицы предсказаний (реальное/фейковое изображение)
        adv_criterion: функция потерь для состязательного обучения; принимает предсказания дискриминатора и истинные метки,
                       возвращая состязательную ошибку (её необходимо минимизировать)
        identity_criterion: функция потерь реконструкции, используемая для потерь идентичности и циклической согласованности;
                            принимает два набора изображений и возвращает их разницу по пикселям (её необходимо минимизировать)
        cycle_criterion: функция потерь циклической согласованности; принимает реальные изображения X, изображения,
                         прошедшие через генератор X->Y и затем Y->X, и возвращает ошибку циклической согласованности
                         (её необходимо минимизировать). На практике cycle_criterion == identity_criterion == L1 loss.
        lambda_identity: коэффициент для взвешивания потерь идентичности
        lambda_cycle: коэффициент для взвешивания потерь циклической согласованности
    '''
    # Должны быть учтены оба направления — генераторы работают совместно.
    # Должны быть испоьзованы коэффициенты lambda для потерь идентичности и циклической согласованности.

    # Состязательная потеря (Adversarial Loss)
    adv_loss_BA, fake_A = get_gen_adversarial_loss(real_B, disc_A, gen_BA, adv_criterion)
    adv_loss_AB, fake_B = get_gen_adversarial_loss(real_A, disc_B, gen_AB, adv_criterion)
    gen_adversarial_loss = adv_loss_BA + adv_loss_AB

    # Потеря идентичности (Identity Loss)
    identity_loss_A, identity_A = get_identity_loss(real_A, gen_BA, identity_criterion)
    identity_loss_B, identity_B = get_identity_loss(real_B, gen_AB, identity_criterion)
    gen_identity_loss = identity_loss_A + identity_loss_B

    # Потеря циклической согласованности (Cycle-consistency Loss)
    cycle_loss_BA, cycle_A = get_cycle_consistency_loss(real_A, fake_B, gen_BA, cycle_criterion)
    cycle_loss_AB, cycle_B = get_cycle_consistency_loss(real_B, fake_A, gen_AB, cycle_criterion)
    gen_cycle_loss = cycle_loss_BA + cycle_loss_AB

    # Итоговая функция потерь
    gen_loss = lambda_identity * gen_identity_loss + lambda_cycle * gen_cycle_loss + gen_adversarial_loss
    return gen_loss, fake_A, fake_B


from skimage import color
import numpy as np

plt.rcParams["figure.figsize"] = (10, 10)



if __name__ == "__main__":
    pass