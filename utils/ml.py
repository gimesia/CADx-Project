from torch import cat
import logging
import pandas as pd
from sklearn.base import ClassifierMixin

from utils.feature_extraction import FeatureExtractionStrategy
from utils.loader import FactoryLoader
from utils.preprocessing import PreprocessingFactory

# Set up logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLPipeline:
    def __init__(self, dataset_path, preprocessing_factory: PreprocessingFactory,
                 feature_strategy: FeatureExtractionStrategy,
                 classifiers: list[ClassifierMixin], percentage: int = 100,
                 verbose: bool = False, shuffle=False):
        self.loader = FactoryLoader(path=dataset_path, factory=preprocessing_factory, percentage=percentage,
                                    shuffle=shuffle)
        self.feature_strategy = feature_strategy
        self.classifiers = classifiers
        self.feature_matrix = None
        self.labels = None
        self.is_extracted = False
        self.fitted_classifiers = {}
        self.verbose = verbose  # Control logging verbosity

        if self.verbose:
            logger.info("MLPipeline initialized with dataset path: %s", dataset_path)
            logger.info("Preprocessing steps", self.loader.get_transformation_steps())

    def run_feature_extraction(self):
        """Extracts features and labels from the dataset."""
        if self.verbose:
            logger.info("Running feature extraction...")

        loader_data = self.loader.get_loader()  # This returns the dataset loader
        self.feature_matrix = self.feature_strategy.run(loader_data)
        # Extract labels from batches of tensors and convert them into a flat array
        labels_list = []
        for batch in loader_data:
            _, labels_batch = batch  # Assuming each batch is (features, labels)
            labels_list.append(labels_batch)

        # If using PyTorch, convert the list of tensors to a single tensor and then to a NumPy array
        self.labels = cat(labels_list).numpy()

        self.is_extracted = True

        if self.verbose:
            logger.info("Feature extraction completed. Extracted %d features.", len(self.feature_matrix))

    def fit_classifiers(self):
        """
        Fits all classifiers on the extracted features using the labels obtained from the loader.
        """
        if not self.is_extracted:
            raise RuntimeError("Features must be extracted before fitting classifiers.")

        if self.verbose:
            logger.info("Fitting classifiers...")

        self.fitted_classifiers = {}
        for clf in self.classifiers:
            print(self.labels)
            clf.fit(self.feature_matrix, self.labels)
            self.fitted_classifiers[clf.__class__.__name__] = clf
            if self.verbose:
                logger.info("Fitted classifier: %s", clf.__class__.__name__)

        if self.verbose:
            logger.info("All classifiers have been fitted.")

    def predict_with_classifiers(self, new_dataset_path, percentage=100):
        """
        Predicts the output for a new dataset using all fitted classifiers.

        Args:
        new_dataset_path: The path to the new dataset for prediction.

        Returns:
        A dictionary containing the predictions from all classifiers.
        """
        if not self.fitted_classifiers:
            raise RuntimeError("Classifiers must be fitted before making predictions.")

        if self.verbose:
            logger.info("Predicting with classifiers on dataset: %s", new_dataset_path)

        # Load and extract features from the new dataset
        new_loader = FactoryLoader(path=new_dataset_path, factory=self.loader.get_factory(), percentage=percentage)
        new_feature_matrix = self.feature_strategy.run(new_loader.get_loader())

        # Store predictions in a dictionary
        predictions = {}
        for clf_name, clf in self.fitted_classifiers.items():
            predictions[clf_name] = clf.predict(new_feature_matrix)
            if self.verbose:
                logger.info("Predictions made with classifier: %s", clf_name)

        return predictions

    def get_feature_names(self):
        """Returns feature names extracted by the strategy."""
        return self.feature_strategy.get_feature_names()

    def get_classes(self):
        """Returns classes found in training."""
        return self.loader.get_classes()

    def save_feature_matrix_to_excel(self, output_dir: str = './'):
        """
        Saves the feature matrix to an Excel file using preprocessing step names in the filename.

        Args:
        output_dir (str): Directory where the Excel file will be saved. Defaults to the current directory.

        Returns:
        str: The path to the saved Excel file.
        """
        if not self.is_extracted:
            raise RuntimeError("Features must be extracted before saving.")

        if self.verbose:
            logger.info("Saving feature matrix to Excel...")

        # Get the step names from the preprocessing factory
        step_names = "_".join(self.loader.get_transformation_steps().keys())
        if not step_names:
            step_names = 'default'

        # Create the file path using the step names
        file_name = f'features_{step_names}.xlsx'
        file_path = output_dir + file_name

        # Convert the feature matrix to a pandas DataFrame
        feature_df = pd.DataFrame(self.feature_matrix, columns=self.get_feature_names())

        # Save to Excel
        feature_df.to_excel(file_path, index=False)

        if self.verbose:
            logger.info("Feature matrix saved to %s", file_path)

        return file_path

# Example usage:
# pipeline = MLPipeline(dataset_path="data/", preprocessing_factory=some_factory, feature_strategy=strategy, classifiers=[clf1, clf2], verbose=True)
# pipeline.run_feature_extraction()
# pipeline.fit_classifiers()
# predictions = pipeline.predict_with_classifiers(new_dataset_path="new_data/")
# pipeline.save_feature_matrix_to_excel(output_dir="./output/")
