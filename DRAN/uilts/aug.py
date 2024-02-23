import random
import numpy as np
from PIL import Image, ImageChops
import imgaug.augmenters as iaa



def AddColor(img):
    print(type(img))
    value = random.randint(-40, 40)
    img = np.clip(img+value, a_min=0.0, a_max=255.0)
    return img


def group1_aug_train(img):
    img = np.array(img)
    if np.random.randint(2):
        iaa_seq1 = iaa.Sequential([
                iaa.Add(value=(-40, 40), per_channel=True),  # Add color
                iaa.GammaContrast(gamma=(0.5, 1.5)),  # GammaContrast with a gamma of 0.5 to 1.5
                ])
        img = iaa_seq1(images=img)
    if np.random.randint(2):
        scale = np.random.random()*(1.05-0.9) + 0.9
        translate_percent = np.random.random()*(0.08-(-0.08)) + (-0.08)
        rotate = np.random.random()*(15-(-15)) + (-15)
        iaa_seq2 = iaa.Sequential([
                # Scale images、move image、rotate image
                iaa.Affine(mode="reflect", scale=scale, translate_percent=translate_percent, rotate=rotate),
            ])
        img = iaa_seq2(images=img)
    
    img = Image.fromarray(img)
    return img