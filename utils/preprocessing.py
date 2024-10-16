import numpy as np
import torch
import cv2


# Define the base class for preprocessing steps
class PreprocessingStep:
    def apply(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError("This method should be overridden in the derived class.")

    def get_step_params(self):
        raise NotImplementedError("This method should be overridden in the derived class.")


# Implement concrete preprocessing steps
class Pad2Square(PreprocessingStep):
    def __init__(self, fill=0):
        self.fill = fill

    def apply(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        if height == width:
            return image  # Image is already square, no need to pad

        size = max(height, width)
        delta_w = size - width
        delta_h = size - height
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        color = [self.fill] * 3  # Assuming image has 3 channels (RGB)
        padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return padded_image

    def get_step_params(self):
        return {"name": "pad2square", "fill": self.fill}


class Smoothing(PreprocessingStep):
    def __init__(self, kernel_size=5):
        self.kernel_size = kernel_size

    def apply(self, image: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(image, (self.kernel_size, self.kernel_size), 0)

    def get_step_params(self):
        return {"name": "smoothing", "kernel_size": self.kernel_size}


class HairRemoval(PreprocessingStep):
    def apply(self, image: np.ndarray) -> np.ndarray:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, binary_mask = cv2.threshold(gray_image, 10, 255, cv2.THRESH_BINARY_INV)
        inpainted_image = cv2.inpaint(image, binary_mask, 3, cv2.INPAINT_TELEA)
        return inpainted_image

    def get_step_params(self):
        return {"name": "hair_removal"}


class Resize(PreprocessingStep):
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size

    def apply(self, image: np.ndarray) -> np.ndarray:
        return cv2.resize(image, self.target_size)

    def get_step_params(self):
        return {"name": "resize", "size": self.target_size}

class StdNormalize(PreprocessingStep):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def apply(self, image: np.ndarray) -> np.ndarray:
        image = image / 255.0
        image = (image - self.mean) / self.std
        return image

    def get_step_params(self):
        return {"name": "std_norm", "mean": self.mean, "std": self.std}

class FloatNormalize(PreprocessingStep):
    def __init__(self, ):
        pass

    def apply(self, image: np.ndarray) -> np.ndarray:
        image = image / 255.0
        return image

    def get_step_params(self):
        return {"name": "01_norm"}

# Factory class to add and manage preprocessing steps
class PreprocessingFactory:
    def __init__(self):
        self.steps = []

    def pad2square(self, fill=0):
        self.steps.append(Pad2Square(fill))

    def gaussian_smoothing(self, kernel_size=5):
        self.steps.append(Smoothing(kernel_size=kernel_size))

    def hair_removal(self):
        self.steps.append(HairRemoval())

    def resize(self, target_size=(224, 224)):
        self.steps.append(Resize(target_size=target_size))

    def std_normalize(self, mean, std):
        self.steps.append(StdNormalize(mean=mean, std=std))

    def normalize2float(self):
        self.steps.append(FloatNormalize())

    def apply_preprocessing(self, image: np.ndarray) -> np.ndarray:
        for step in self.steps:
            image = step.apply(image)
        return image

    def get_steps(self):
        steps = ([i.get_step_params() for i in self.steps]) # get steps with params
        grouped_dict = {}
        for step in steps:
            name = step.get('name') # get name of the step
            grouped_dict[name] = step # add name of step as key in output dict
            del(grouped_dict[name]["name"]) # delete 'name' key
        return grouped_dict


# PreprocessMelanoma class to apply the preprocessing pipeline
class PreprocessMelanoma:
    def __init__(self, factory: PreprocessingFactory):
        self.factory = factory

    def __call__(self, image):
        # Convert the Image to numpy array
        image = np.array(image)

        # Apply the factory's preprocessing steps
        image = self.factory.apply_preprocessing(image)

        # Convert numpy array to a PyTorch tensor with shape (C, H, W)
        image = torch.tensor(image).permute(2, 0, 1).float()

        return image
