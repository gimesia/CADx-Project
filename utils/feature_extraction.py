import numpy as np
from torch.utils.data import DataLoader
from skimage.feature import local_binary_pattern, hog
from skimage.feature import graycomatrix, graycoprops
import cv2

# Base FeatureExtractor class (Strategy Pattern)
class FeatureExtractor:
    def __init__(self, name: str, threshold: float = 0.1):
        self.name = name
        self.threshold = threshold
        
    def extract(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError("This method should be implemented by subclasses.")

    def get_feature_name(self) -> list:
        """Return the names of the features extracted by this extractor."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def apply_threshold_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Applies a binary threshold mask if a significant portion of the image is black (background).
        Only regions with intensity greater than 10 are considered in the ROI.
        """
        # Ensure the image is either grayscale (1 channel) or RGB (3 channels)
        if image.ndim == 3 and image.shape[-1] == 3:  # RGB image (H, W, 3)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif image.ndim == 2:  # Grayscale image
            gray = image
        else:
            raise ValueError(f"Unexpected number of channels: {image.shape[-1]}. Expected 1 or 3 channels.")

        # Calculate the black pixel ratio
        black_pixels = np.sum(gray == 0)
        total_pixels = gray.size
        black_pixel_ratio = black_pixels / total_pixels

        #print(f"Black pixel ratio: {black_pixel_ratio}")

        if black_pixel_ratio > self.threshold:
            # Create a binary mask where non-black regions are preserved
            _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

            # Ensure both the mask and image are the same type and size
            mask = mask.astype(np.uint8)  # Ensure the mask is uint8
            if image.ndim == 3:
                image = image.astype(np.uint8)  # Ensure the image is uint8

            # Check if the mask and the image match in size
            if mask.shape != image.shape[:2]:
                raise ValueError(f"Mask size {mask.shape} and image size {image.shape[:2]} do not match.")

            # Apply mask to the image (only works with matching sizes and types)
            masked_image = cv2.bitwise_and(image, image, mask=mask)
            #print("Mask applied. Extracting from the region of interest.")
            return masked_image
        else:
            #print("No significant black regions. Extracting features from the entire image.")
            return image


# Specific feature extraction strategies
class MeanExtractor(FeatureExtractor):
    def __init__(self, threshold=0.1):
        super().__init__(name="mean", threshold=threshold)

    def extract(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 3 and image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0)) 
        masked_image = self.apply_threshold_mask(image)
        return np.mean(masked_image, axis=(0, 1))  # Compute mean across height, width

    def get_feature_name(self) -> list:
        return [f"{self.name}_channel_{i}" for i in range(3)]


class StdExtractor(FeatureExtractor):
    def __init__(self, threshold=0.1):
        super().__init__(name="std", threshold=threshold)

    def extract(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 3 and image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0)) 
        masked_image = self.apply_threshold_mask(image)
        return np.std(masked_image, axis=(0, 1))  # Compute std across height, width

    def get_feature_name(self) -> list:
        return [f"{self.name}_channel_{i}" for i in range(3)]


class VarExtractor(FeatureExtractor):
    def __init__(self, threshold=0.1):
        super().__init__(name="var", threshold=threshold)

    def extract(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 3 and image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0)) 
        masked_image = self.apply_threshold_mask(image)
        return np.var(masked_image, axis=(0, 1))  # Compute variance across height, width

    def get_feature_name(self) -> list:
        return [f"{self.name}_channel_{i}" for i in range(3)]


class MaxExtractor(FeatureExtractor):
    def __init__(self, threshold=0.1):
        super().__init__(name="max", threshold=threshold)

    def extract(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 3 and image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0)) 
        masked_image = self.apply_threshold_mask(image)
        return np.max(masked_image, axis=(0, 1))  # Compute max across height, width

    def get_feature_name(self) -> list:
        return [f"{self.name}_channel_{i}" for i in range(3)]

class LBPExtractor(FeatureExtractor):
    def __init__(self, radius=1, n_points=8, threshold=0.1):
        super().__init__(name="lbp", threshold=threshold)
        self.radius = radius
        self.n_points = n_points

    def extract(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 3 and image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))   # Transpose to HWC format
        if image.ndim == 3 and image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Apply the threshold-based mask
        masked_image = self.apply_threshold_mask(image)

        # Replace masked regions with zeros (or another constant)
        masked_image[masked_image == 0] = np.nan  # Handle NaNs as invalid areas
        
        # Apply LBP
        lbp = local_binary_pattern(np.nan_to_num(masked_image), self.n_points, self.radius, method='uniform')

        # Compute LBP histogram
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, self.n_points + 3), range=(0, self.n_points + 2))

        # Normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-6)

        return hist

    def get_feature_name(self) -> list:
        return [f"{self.name}_bin_{i}" for i in range(self.n_points + 2)]

class GaborExtractor(FeatureExtractor):
    def __init__(self, num_orientations=8, num_scales=4, threshold=0.1):
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
                filtered_images.append(np.mean(filtered_img))  # You can choose other stats like variance

        return np.array(filtered_images)

    def get_feature_name(self) -> list:
        return [f"{self.name}_filter_{i}" for i in range(self.num_orientations * self.num_scales)]
    
class GLCMExtractor(FeatureExtractor):
    def __init__(self, distances=[1], angles=[0], properties=['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM'], threshold=0.1):
        super().__init__(name="glcm", threshold=threshold)
        self.distances = distances
        self.angles = angles
        self.properties = properties

    def extract(self, image: np.ndarray) -> np.ndarray:
        # Convert to grayscale if image has multiple channels
        if image.ndim == 3 and image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))  # Convert to HWC format
        if image.ndim == 3 and image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Apply threshold mask
        masked_image = self.apply_threshold_mask(image)
        
        # Compute GLCM matrix and texture features
        glcm = graycomatrix(masked_image.astype(np.uint8), 
                            distances=self.distances, 
                            angles=self.angles, 
                            symmetric=True, 
                            normed=True)

        # Extract specified properties
        feature_values = []
        for prop in self.properties:
            feature_values.extend(graycoprops(glcm, prop).flatten())
        
        return np.array(feature_values)

    def get_feature_name(self) -> list:
        feature_names = []
        for prop in self.properties:
            for dist in self.distances:
                for angle in self.angles:
                    feature_names.append(f"{self.name}_{prop}_dist_{dist}_angle_{angle}")
        return feature_names

# Feature extraction pipeline
class FeatureExtractionStrategy:
    def __init__(self):
        self.extractors = []
        self.feature_vectors = None

    def add_extractor(self, extractor: FeatureExtractor):
        self.extractors.append(extractor)

    def extract_features(self, image: np.ndarray) -> np.ndarray:
        # Apply all the extractors and concatenate their results
        #print(image.shape)
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

            for batch in loader:
                #print("Batch contents: ", batch)
                
                images, labels = batch
                #print(labels)
                # Iterate through images in the batch
                for image, label in zip(images, labels):
                    image_np = image.numpy()  # Convert tensor to NumPy
                    feature_vector = self.extract_features(image_np)

                    # Add feature vector and corresponding label to the lists
                    feature_vectors.append(feature_vector)
                    labels_list.append(label.item())  # Convert label tensor to scalar

            # Convert to NumPy arrays
            feature_matrix = np.array(feature_vectors)
            labels_array = np.array(labels_list)
            
            self.feature_vectors = feature_matrix
            return feature_matrix, labels_array


# The extract_features function will now use the feature extraction pipeline
def extract_features(loader: DataLoader, feature_extraction: FeatureExtractionStrategy):
    feature_vectors = []

    for batch in loader:
        images, labels = batch

        # Iterate through images in the batch
        for image in images:
            image_np = image.numpy()  # Convert tensor to NumPy

            # Extract features using the provided pipeline
            feature_vector = feature_extraction.extract_features(image_np)

            # Add to feature vector list
            feature_vectors.append(feature_vector)

    # Convert list of feature vectors to a numpy array
    feature_matrix = np.array(feature_vectors)

    return feature_matrix