import numpy as np
import torch
import cv2
from skimage.color import rgb2gray

from utils.utils import norm


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

class CLAHE(PreprocessingStep):
    def __init__(self, clip_limit = 2.0, tile_grid_size = (8,8)):

        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def apply(self, image: np.ndarray) -> np.ndarray:

        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

        l_channel, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit= self.clip_limit, tileGridSize=self.tile_grid_size)

        cl = clahe.apply(l_channel)

        lab_clahe = cv2.merge((cl, a, b))

        image_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)

        return image_clahe
    
    def get_step_params(self):
        return {"name": "clahe", "clip_limit": self.clip_limit, "tile_grid_size": self.tile_grid_size}


class Smoothing(PreprocessingStep):
    def __init__(self, kernel_size=5):
        self.kernel_size = kernel_size

    def apply(self, image: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(image, (self.kernel_size, self.kernel_size), 0)

    def get_step_params(self):
        return {"name": "smoothing", "kernel_size": self.kernel_size}


def create_tilted_structuring_elements(width, height, n):
    if width % 2 == 0 or height % 2 == 0:
        raise ValueError("Structuring element dimensions must be odd.")

    base = np.zeros((width, width), dtype=np.uint8)
    for k in range(width // 2 - height // 2, width // 2 + height // 2 + 1):
        base = cv2.line(base, (0, k), (width, k), 255)

    SEs = [base]
    angle_step = 180.0 / n
    for k in range(1, n):
        M = cv2.getRotationMatrix2D((base.shape[1] / 2.0, base.shape[0] / 2.0), k * angle_step, 1.0)
        SE = cv2.warpAffine(base, M, (width, width), flags=cv2.INTER_NEAREST)
        SEs.append(SE)

    return SEs


class HairRemoval(PreprocessingStep):
    def __init__(self, kernel_size=3, directions=8, length=3):
        self.kernel_size = kernel_size
        self.directions = directions
        self.length = length

    def apply(self, image: np.ndarray) -> np.ndarray:
        img = norm(image, np.uint8)
        img_gray = norm(rgb2gray(img), np.uint8)

        SEs = create_tilted_structuring_elements(self.length, 1, self.directions)
        sum_blackhats = np.zeros(img_gray.shape, np.uint16)

        for i, SE in enumerate(SEs):
            blackhat = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, SE)
            sum_blackhats = cv2.add(sum_blackhats, blackhat.astype(np.uint16))

        sum_blackhats = cv2.normalize(sum_blackhats, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        _, thresholded = cv2.threshold(sum_blackhats, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        thresholded = cv2.dilate(
            thresholded,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.kernel_size, self.kernel_size))
        )
        result = cv2.inpaint(img, thresholded, 5, cv2.INPAINT_TELEA)

        return result

    def get_step_params(self):
        return {"name": "hair_removal",
                "kernel_size": self.kernel_size,
                "directions": self.directions,
                "length": self.length}


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

class CropROI(PreprocessingStep):
    def __init__(self, black_threshold_ratio=0.1):
        self.black_threshold_ratio = black_threshold_ratio  # Threshold for black pixel ratio

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Crops the region of interest (ROI) by removing large black areas around the image,
        but only if the black pixel ratio exceeds the threshold.
        """
        # Convert image to grayscale for thresholding
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Check for black pixel regions (assume black pixels have value close to 0)
        black_pixels = np.sum(gray == 0)
        total_pixels = gray.size
        black_pixel_ratio = black_pixels / total_pixels

        # Check if black pixels exceed the threshold
        if black_pixel_ratio > self.black_threshold_ratio:
            #print(f"Black pixel ratio {black_pixel_ratio:.2f} exceeds threshold. Applying cropping.")

            # Apply a binary threshold to find regions that are not black
            _, binary_mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

            # Find contours on the binary mask
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # Find the largest contour (assumed to be the region of interest)
                largest_contour = max(contours, key=cv2.contourArea)

                # Get the bounding box for the largest contour
                x, y, w, h = cv2.boundingRect(largest_contour)

                # Crop the image to this bounding box
                cropped_image = image[y:y+h, x:x+w]
                return cropped_image
            else:
                #print("No significant region of interest found. Returning the original image.")
                return image
        else:
            #print(f"Black pixel ratio {black_pixel_ratio:.2f} is below threshold. No cropping applied.")
            return image

    def get_step_params(self):
        return {"name": "crop_roi", "black_threshold_ratio": self.black_threshold_ratio}
    
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

    def crop_roi(self):
        """Add the cropping step to the pipeline."""
        self.steps.append(CropROI())

    def pad2square(self, fill=np.nan):
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

    def clahe(self, clip_limit = 2.0, tile_grid_size=(8,8)):
        self.steps.append(CLAHE(clip_limit=clip_limit, tile_grid_size=tile_grid_size))

    def add_preprocessing_step(self, step: PreprocessingStep):
        self.steps.append(step)

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
