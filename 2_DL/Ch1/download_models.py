import torch
from torchvision.io import read_image
from torchvision.models import (
    resnet50, ResNet50_Weights,
    resnet152, ResNet152_Weights,
    resnet101, ResNet101_Weights,
    wide_resnet101_2, Wide_ResNet101_2_Weights,
    efficientnet_v2_m, EfficientNet_V2_M_Weights,
    vgg19_bn, VGG19_BN_Weights,
    convnext_base, ConvNeXt_Base_Weights
)

# Read the image file into a Tensor
img = read_image(r"C:\Users\gimes\Downloads\Grace_Hopper.jpg")

# Helper function to run inference
def run_inference(model, weights, img):
    model.eval()
    preprocess = weights.transforms()
    batch = preprocess(img).unsqueeze(0)
    prediction = model(batch).squeeze(0).softmax(0)
    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    category_name = weights.meta["categories"][class_id]
    return f"{category_name}: {100 * score:.1f}%"

# Models and their weights
models_and_weights = [
    (resnet50, ResNet50_Weights.DEFAULT, "ResNet50"),
    (resnet152, ResNet152_Weights.IMAGENET1K_V2, "ResNet152"),
    (resnet101, ResNet101_Weights.IMAGENET1K_V2, "ResNet101"),
    (wide_resnet101_2, Wide_ResNet101_2_Weights.IMAGENET1K_V2, "Wide ResNet101_2"),
    (efficientnet_v2_m, EfficientNet_V2_M_Weights.IMAGENET1K_V1, "EfficientNet_V2_M"),
    (vgg19_bn, VGG19_BN_Weights.IMAGENET1K_V1, "VGG19_BN"),
    # (convnext_base, ConvNeXt_Base_Weights.IMAGENET1K_V1, "ConvNeXt_Base")
]

# Run inference for all models
for model_fn, weights, model_name in models_and_weights:
    model = model_fn(weights=weights)
    result = run_inference(model, weights, img)
    print(f"{model_name} - {result}")

