import numpy as np
from scipy import ndimage
from skimage.color import rgb2gray
from torch.utils.data import DataLoader
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
import cv2
from scipy.stats import skew, kurtosis, iqr, entropy
from utils.utils import norm

TH = 0.05

# Base FeatureExtractor class (Strategy Pattern)
class FeatureExtractor:
    def __init__(self, name: str, threshold: float = TH, color_space=None):
        self.name = name
        self.threshold = threshold
        self.color_space = color_space

    def extract(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError("This method should be implemented by subclasses.")

    def get_feature_name(self) -> list:
        """Return the names of the features extracted by this extractor."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def convert_color_space(self, image: np.ndarray) -> np.ndarray:
        """
        Convert the image to the specified color space.
        """
        conversion = {
            'hsv': cv2.COLOR_RGB2HSV,
            'lab': cv2.COLOR_RGB2LAB,
            'gray': cv2.COLOR_RGB2GRAY
        }
        if self.color_space is not None:
            return cv2.cvtColor(image,
                                conversion.get(self.color_space, None)) if self.color_space in conversion else image
        return image

    def apply_threshold_mask(self, image: np.ndarray, fill_nan=True) -> np.ndarray:
        """Applies a binary threshold mask if a significant portion of the image is black (background)."""
        if image.ndim == 3 and image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))  # Transpose to HWC format

        if image.ndim == 3 and image.shape[-1] == 3:  # RGB image (H, W, 3)
            gray = norm(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), np.uint8)
        elif image.ndim == 2:  # Grayscale image
            gray = norm(image, np.uint8)
        else:
            raise ValueError(f"Unexpected number of channels: {image.shape[-1]}. Expected 1 or 3 channels.")

        _, mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        mask = ndimage.binary_fill_holes(mask)

        # Calculate the black pixel ratio
        black_pixels = np.sum(mask == 0)
        total_pixels = gray.size
        black_pixel_ratio = black_pixels / total_pixels


        if black_pixel_ratio > self.threshold:
            # Create a binary mask where non-black regions are preserved
            mask = mask.astype(np.uint8)
            # if image.ndim == 3:
            # image = image.astype(np.uint8)

            # Check if the mask and the image match in size
            if mask.shape != image.shape[:2]:
                raise ValueError(f"Mask size {mask.shape} and image size {image.shape[:2]} do not match.")

            # Apply mask to the image (only works with matching sizes and types)
            masked_image = image.copy()

            masked_image[mask == 0] = np.nan if fill_nan else 0

            return masked_image
        else:
            return image


# Specific feature extraction strategies
class MeanExtractor(FeatureExtractor):
    def __init__(self, color_space='rgb', threshold=0.1):
        super().__init__(name="mean", threshold=threshold)
        self.color_space = color_space

    def extract(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 3 and image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))  # Convert to HWC format

        image = self.convert_color_space(image)
        masked_image = self.apply_threshold_mask(image)
        return np.mean(masked_image, axis=(0, 1))

    def get_feature_name(self) -> list:
        return [f"{self.name}_{self.color_space}_channel_{i}" for i in range(3)]


class StdExtractor(FeatureExtractor):
    def __init__(self, color_space='rgb', threshold=TH):
        super().__init__(name="std", threshold=threshold)
        self.color_space = color_space

    def extract(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 3 and image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))  # Convert to HWC format

        image = self.convert_color_space(image)
        masked_image = self.apply_threshold_mask(image)
        return np.std(masked_image, axis=(0, 1))

    def get_feature_name(self) -> list:
        return [f"{self.name}_{self.color_space}_channel_{i}" for i in range(3)]


class VarExtractor(FeatureExtractor):
    def __init__(self, color_space='rgb', threshold=0.1):
        super().__init__(name="var", threshold=threshold)
        self.color_space = color_space

    def extract(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 3 and image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))  # Convert to HWC format

        image = self.convert_color_space(image)
        masked_image = self.apply_threshold_mask(image)
        return np.var(masked_image, axis=(0, 1))

    def get_feature_name(self) -> list:
        return [f"{self.name}_{self.color_space}_channel_{i}" for i in range(3)]


class MaxExtractor(FeatureExtractor):
    def __init__(self, color_space='rgb', threshold=0.1):
        super().__init__(name="max", threshold=threshold)
        self.color_space = color_space

    def extract(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 3 and image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))  # Convert to HWC format

        image = self.convert_color_space(image)
        masked_image = self.apply_threshold_mask(image)
        return np.max(masked_image, axis=(0, 1))

    def get_feature_name(self) -> list:
        return [f"{self.name}_{self.color_space}_channel_{i}" for i in range(3)]


class GradientExtractor(FeatureExtractor):
    def __init__(self, threshold=0.1):
        super().__init__(name="gradient", threshold=threshold)

    def extract(self, image: np.ndarray) -> np.ndarray:
        # Convert to grayscale if image has multiple channels
        if image.ndim == 3 and image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))  # Convert to HWC format

        if image.ndim == 3 and image.shape[-1] == 3:
            gray = rgb2gray(image)
        elif image.ndim == 2:
            gray = image
        else:
            print(image.ndim)
            raise ValueError("Unexpected number of channels")

        masked_image = self.apply_threshold_mask(gray)

        grad_x = cv2.Sobel(masked_image, cv2.CV_64F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(masked_image, cv2.CV_64F, 0, 1, ksize=5)

        gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        gradient_direction = np.arctan2(grad_y, grad_x)

        return np.array([np.mean(gradient_magnitude), np.std(gradient_magnitude),
                         np.mean(gradient_direction), np.std(gradient_direction)])

    def get_feature_name(self) -> list:
        return [f"{self.name}_magnitude_mean", f"{self.name}_magnitude_std",
                f"{self.name}_direction_mean", f"{self.name}_direction_std"]


class LBPExtractor(FeatureExtractor):
    def __init__(self, radius=1, n_points=8, threshold=TH):
        super().__init__(name="lbp", threshold=threshold)
        self.radius = radius
        self.n_points = n_points

    def extract(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 3 and image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))  # Transpose to HWC format

        if image.ndim == 3 and image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Apply the threshold-based mask
        masked_image = self.apply_threshold_mask(image)

        # Replace masked regions with zeros (or another constant)
        masked_image[masked_image == 0] = np.nan  # Handle NaNs as invalid areas TODO!

        masked_image = norm(masked_image, np.uint8)

        # Apply LBP
        lbp = local_binary_pattern(np.nan_to_num(masked_image), self.n_points, self.radius, method='uniform')

        # Compute LBP histogram
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, self.n_points + 3), range=(0, self.n_points + 2))

        # Normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-6)

        return hist

    def get_feature_name(self) -> list:
        return [f"{self.name}_rad{self.radius}_bins{self.n_points}_{i}" for i in range(self.n_points + 2)]


class MultiScaleLBPExtractor(FeatureExtractor):
    def __init__(self, scales=((8, 1), (16, 2), (24, 3)), threshold=TH):
        super().__init__(name="multiscale_lbp", threshold=threshold)
        self.scales = scales

    def extract(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 3 and image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))  # Transpose to HWC format

        if image.ndim == 3 and image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Apply the threshold-based mask
        masked_image = self.apply_threshold_mask(image)

        # Replace masked regions with zeros (or another constant)
        masked_image[masked_image == 0] = np.nan  # Handle NaNs as invalid areas TODO!

        masked_image = norm(masked_image, np.uint8)

        # Compute LBP histograms for each scale and concatenate them
        histograms = []
        for n_points, radius in self.scales:
            lbp = local_binary_pattern(np.nan_to_num(masked_image), n_points, radius, method='uniform')
            hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
            hist = hist.astype("float")
            hist /= (hist.sum() + 1e-6)  # Normalize histogram
            histograms.extend(hist)  # Append to overall feature vector

        return np.array(histograms)

    def get_feature_name(self) -> list:
        feature_names = []
        for n_points, radius in self.scales:
            feature_names.extend([f"{self.name}_rad{radius}_bins{n_points}_{i}" for i in range(n_points + 2)])
        return feature_names


class GaborExtractor(FeatureExtractor):
    def __init__(self, num_orientations=8, num_scales=4, threshold=TH):
        super().__init__(name="gabor", threshold=threshold)
        self.num_orientations = num_orientations
        self.num_scales = num_scales

    def extract(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 3 and image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))  # Convert to HWC

        if image.ndim == 3 and image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # Convert to grayscale

        # Apply Gabor filters
        filtered_images = []
        for theta in np.linspace(0, np.pi, self.num_orientations):
            for scale in np.linspace(5, 20, self.num_scales):
                kernel = cv2.getGaborKernel((21, 21), scale, theta, 10, 0.5, 0, ktype=cv2.CV_32F)
                filtered_img = cv2.filter2D(image, cv2.CV_8UC3, kernel)
                filtered_images.append(np.mean(filtered_img))

        return np.array(filtered_images)

    def get_feature_name(self) -> list:
        return [f"{self.name}_filter_{i}" for i in range(self.num_orientations * self.num_scales)]


class GLCMExtractor(FeatureExtractor):
    def __init__(self, distances=[1], angles=[0],
                 properties=['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM'],
                 threshold=TH):
        super().__init__(name="glcm", threshold=threshold)
        self.distances = distances
        self.angles = angles
        self.properties = properties

    def extract(self, image: np.ndarray) -> np.ndarray:
        # Convert to grayscale if the image has multiple channels
        if image.ndim == 3 and image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))  # Convert to HWC format

        if image.ndim == 3 and image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Apply threshold mask
        masked_image = self.apply_threshold_mask(image)

        # Normalize to int
        masked_image = norm(masked_image, np.uint8)

        # Compute GLCM matrix
        glcm = graycomatrix(masked_image,
                            distances=self.distances, 
                            angles=self.angles, 
                            symmetric=True, 
                            normed=True)

        # Extract specified properties
        feature_values = []
        for prop in self.properties:
            if prop == 'entropy':
                # Calculate entropy manually for each distance and angle
                entropy_values = -np.sum(glcm * np.log(glcm + 1e-10), axis=(0, 1))  # Add a small epsilon to avoid log(0)
                feature_values.extend(entropy_values.flatten())
            else:
                feature_values.extend(graycoprops(glcm, prop).flatten())

        return np.array(feature_values)

    def get_feature_name(self) -> list:
        feature_names = []
        for prop in self.properties:
            for dist in self.distances:
                for angle in self.angles:
                    feature_names.append(f"{self.name}_{prop}_dist_{dist}_angle_{angle}")
        return feature_names


class ColorMomentsExtractor(FeatureExtractor):
    def __init__(self, color_space='rgb', threshold=TH):
        super().__init__(name="color_moments", threshold=threshold)
        self.color_space = color_space.lower()

    def extract(self, image: np.ndarray) -> np.ndarray:
        # Ensure the image is in HWC format if it's in CHW
        if image.ndim == 3 and image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))  # Convert to HWC format

        # Convert the image to the specified color space
        image = self.convert_color_space(image)

        # Apply the threshold-based mask
        masked_image = self.apply_threshold_mask(image)

        # Compute additional color moments for each channel
        color_moments = []
        for i in range(masked_image.shape[-1]):  # Iterate over each color channel
            channel = masked_image[:, :, i]
            mean_val = np.nanmean(channel)
            std_val = np.nanstd(channel)
            skew_val = skew(channel.flatten(), nan_policy='omit')
            kurtosis_val = kurtosis(channel.flatten(), nan_policy='omit')
            median_val = np.nanmedian(channel)
            var_val = np.nanvar(channel)
            min_val = np.nanmin(channel)
            max_val = np.nanmax(channel)
            iqr_val = iqr(channel.flatten(), nan_policy='omit')
            entropy_val = entropy(np.histogram(channel, bins=256)[0] + 1e-6)

            color_moments.extend([mean_val, std_val, skew_val, kurtosis_val,
                                  median_val, var_val, min_val, max_val, iqr_val, entropy_val])

        return np.array(color_moments)

    def get_feature_name(self) -> list:
        # Define channel names based on the color space
        channel_names = ['Channel1', 'Channel2', 'Channel3']
        if self.color_space == 'rgb':
            channel_names = ['R', 'G', 'B']
        elif self.color_space == 'hsv':
            channel_names = ['H', 'S', 'V']
        elif self.color_space == 'lab':
            channel_names = ['L', 'A', 'B']

        # Moments
        moments = ["mean", "std", "skew", "kurtosis", "median", "var", "min", "max", "iqr", "entropy"]
        feature_names = [f"{self.name}_{self.color_space}_{ch}_{moment}" for ch in channel_names for moment in moments]

        return feature_names

# Feature extraction pipeline
class FeatureExtractionStrategy:
    def __init__(self):
        self.extractors = []
        self.feature_vectors = None

    def add_extractor(self, extractor: FeatureExtractor):
        self.extractors.append(extractor)

    def extract_features(self, image: np.ndarray) -> np.ndarray:
        features = [extractor.extract(image) for extractor in self.extractors]
        return np.concatenate(features)

    def get_feature_names(self) -> list:
        # Create a list of feature names from all extractors
        feature_names = []
        for extractor in self.extractors:
            feature_names.extend(extractor.get_feature_name())
        return feature_names

    def run(self, loader: DataLoader):
        feature_vectors = []
        labels_list = []

        total_batches = len(loader)
        batch_count = 0

        for batch in loader:
            images, labels = batch
            batch_count += 1

            for image, label in zip(images, labels):
                image_np = image.numpy()  # Convert tensor to NumPy
                feature_vector = self.extract_features(image_np)

                # Add feature vector and corresponding label to the lists
                feature_vectors.append(feature_vector)
                labels_list.append(label.item())  # Convert label tensor to scalar

            # Log progress every 5 batches
            if batch_count % 5 == 0 or batch_count == total_batches:
                print(f"Processed {batch_count}/{total_batches} batches.")

        feature_matrix = np.array(feature_vectors)
        labels_array = np.array(labels_list)

        self.feature_vectors = feature_matrix
        return feature_matrix, labels_array
