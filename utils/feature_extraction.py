import numpy as np
from torch.utils.data import DataLoader
from skimage.feature import local_binary_pattern
import cv2 

# Base FeatureExtractor class (Strategy Pattern)
class FeatureExtractor:
    def __init__(self, name: str):
        self.name = name

    def extract(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError("This method should be implemented by subclasses.")

    def get_feature_name(self) -> list:
        """Return the names of the features extracted by this extractor."""
        raise NotImplementedError("This method should be implemented by subclasses.")


# Specific feature extraction strategies
class MeanExtractor(FeatureExtractor):
    def __init__(self):
        super().__init__(name="mean")

    def extract(self, image: np.ndarray) -> np.ndarray:
        return np.mean(image, axis=(1, 2))

    def get_feature_name(self) -> list:
        # Return the feature names for each channel (RGB)
        return [f"{self.name}_channel_{i}" for i in range(3)]


class StdExtractor(FeatureExtractor):
    def __init__(self):
        super().__init__(name="std")

    def extract(self, image: np.ndarray) -> np.ndarray:
        return np.std(image, axis=(1, 2))

    def get_feature_name(self) -> list:
        # Return the feature names for each channel (RGB)
        return [f"{self.name}_channel_{i}" for i in range(3)]


class VarExtractor(FeatureExtractor):
    def __init__(self):
        super().__init__(name="var")

    def extract(self, image: np.ndarray) -> np.ndarray:
        return np.var(image, axis=(1, 2))

    def get_feature_name(self) -> list:
        # Return the feature names for each channel (RGB)
        return [f"{self.name}_channel_{i}" for i in range(3)]


class MaxExtractor(FeatureExtractor):
    def __init__(self):
        super().__init__(name="max")

    def extract(self, image: np.ndarray) -> np.ndarray:
        return np.max(image, axis=(1, 2))

    def get_feature_name(self) -> list:
        # Return the feature names for each channel
        return [f"{self.name}_channel_{i}" for i in range(3)]

class LBPExtractor(FeatureExtractor):
    def __init__(self, radius=1, n_points=8):
        super().__init__(name="lbp")
        self.radius = radius
        self.n_points = n_points

    def extract(self, image: np.ndarray) -> np.ndarray:
        # Convert image to grayscale for LBP computation if it's in RGB
        image = np.transpose(image, (1, 2, 0))
    
        if image.ndim == 3 and image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Compute LBP
        lbp = local_binary_pattern(image, self.n_points, self.radius, method='uniform')

        # Create a histogram of the LBP
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, self.n_points + 3), range=(0, self.n_points + 2))

        # Normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-6)  # Avoid division by zero

        return hist

    def get_feature_name(self) -> list:
        # Return the feature names for the LBP histogram
        return [f"{self.name}_bin_{i}" for i in range(self.n_points + 2)]

# Feature extraction pipeline
class FeatureExtractionPipeline:
    def __init__(self):
        self.extractors = []

    def add_extractor(self, extractor: FeatureExtractor):
        self.extractors.append(extractor)

    def extract_features(self, image: np.ndarray) -> np.ndarray:
        # Apply all the extractors and concatenate their results
        features = [extractor.extract(image) for extractor in self.extractors]
        return np.concatenate(features)

    def get_feature_names(self) -> list:
        # Create a list of feature names from all extractors
        feature_names = []
        for extractor in self.extractors:
            feature_names.extend(extractor.get_feature_name())
        return feature_names


# The extract_features function will now use the feature extraction pipeline
def extract_features(loader: DataLoader, feature_extraction: FeatureExtractionPipeline):
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