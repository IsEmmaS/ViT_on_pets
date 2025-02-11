import cv2
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image


def reshape_transform(tensor, height=14, width=14):
    tensor = tensor.permute(1, 0, 2)
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result
import numpy as np
import cv2

def show_cam_on_image1(image, grayscale_cam, use_rgb=True):
    # 确保 grayscale_cam 是单通道的8位图像
    if grayscale_cam.ndim > 2:
        grayscale_cam = grayscale_cam[:, :, 0]  # 如果是多通道，取第一个通道
    if grayscale_cam.dtype != np.uint8:
        grayscale_cam = (255 * grayscale_cam).astype(np.uint8)  # 转换为8位图像

    # 归一化到 [0, 255]
    grayscale_cam = (grayscale_cam - grayscale_cam.min()) / (grayscale_cam.max() - grayscale_cam.min()) * 255
    grayscale_cam = grayscale_cam.astype(np.uint8)

    colormap = cv2.COLORMAP_JET
    heatmap = cv2.applyColorMap(grayscale_cam, colormap)

    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    visualization = cv2.addWeighted(image, 0.5, heatmap, 0.5, 0)

    return visualization

index_2_class = [
    "Abyssinian",
    "Bengal",
    "Birman",
    "Bombay",
    "British_Shorthair",
    "Egyptian_Mau",
    "Maine_Coon",
    "Persian",
    "Ragdoll",
    "Russian_Blue",
    "Siamese",
    "Sphynx",
    "american_bulldog",
    "american_pit_bull_terrier",
    "basset_hound",
    "beagle",
    "boxer",
    "chihuahua",
    "english_cocker_spaniel",
    "english_setter",
    "german_shorthaired",
    "great_pyrenees",
    "havanese",
    "japanese_chin",
    "keeshond",
    "leonberger",
    "miniature_pinscher",
    "newfoundland",
    "pomeranian",
    "pug",
    "saint_bernard",
    "samoyed",
    "scottish_terrier",
    "shiba_inu",
    "staffordshire_bull_terrier",
    "wheaten_terrier",
    "yorkshire_terrier"
]


def get_cam(image_path, model):
    cam = GradCAM(
        model=model,
        target_layers=[model.blocks[-1].norm1],
        reshape_transform=reshape_transform,
    )

    image = cv2.imread(image_path, 1)[:, :, ::-1]
    shape = image.shape[0:-1]
    image = cv2.resize(image, (224, 224))
    input_tensor = preprocess_image(
        image,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    grayscale_cam = cam(input_tensor, targets=None)[0, :]
    visualization = show_cam_on_image(image /255.0, grayscale_cam, use_rgb=True)
    cv2.resize(visualization, shape)
    
    with torch.no_grad():
        output = model(input_tensor)
    
    _, pred = output.max(1)
    label = index_2_class[pred]
    print(f'Predict result is {label}.')
    cv2.imwrite("GradCAMresult/cam-result.png", visualization)


if __name__ == "__main__":
    from model import VisionTransformer
    model = VisionTransformer(num_classes=37)
    model.load_state_dict(torch.load("best_model.pth", weights_only=True))
    model.eval()
    
    get_cam(image_path='images/Abyssinian_7.jpg', model=model)
    
    

