import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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
        self.predictions = {}
        self.verbose = verbose  # Control logging verbosity

        if self.verbose:
            logger.info("MLPipeline initialized with dataset path: %s", dataset_path)
            logger.info("Preprocessing steps", self.loader.get_transformation_steps())

    def run_feature_extraction(self):
        """Extracts features and labels from the dataset."""
        if self.is_extracted:
            logger.info("Features already extracted")
            return
        if self.verbose:
            logger.info("Running feature extraction...")

        loader_data = self.loader.get_loader()  # This returns the dataset loader
        self.feature_matrix, self.labels = self.feature_strategy.run(loader_data)
        self.is_extracted = True

        if self.verbose:
            logger.info("Feature extraction completed. Extracted %d features.", len(self.feature_matrix))

    def fit_classifiers(self):
        """
        Fits all classifiers on the extracted features using the labels obtained from the loader.
        Logs when the fitting starts and the time taken to complete.
        """
        if not self.is_extracted:
            raise RuntimeError("Features must be extracted before fitting classifiers.")

        if self.verbose:
            logger.info("Fitting classifiers...")

        start_time = time.time()  # Start timing

        self.fitted_classifiers = {}
        for clf in self.classifiers:
            if self.verbose:
                logger.info("Fitting classifier: %s", clf.__class__.__name__)

            clf.fit(self.feature_matrix, self.labels)
            self.fitted_classifiers[clf.__class__.__name__] = clf

            if self.verbose:
                logger.info("Fitted classifier: %s", clf.__class__.__name__)

        end_time = time.time()  # End timing
        duration = end_time - start_time

        if self.verbose:
            logger.info("Fitting completed in %.2f seconds.", duration)

    def fit_classifiers_async(self):
        """
        Fits all classifiers on the extracted features using the labels obtained from the loader asynchronously.
        Logs when the fitting starts and the time taken to complete.
        """
        if not self.is_extracted:
            raise RuntimeError("Features must be extracted before fitting classifiers.")

        if self.verbose:
            logger.info("Fitting classifiers asynchronously...")

        start_time = time.time()  # Start timing

        self.fitted_classifiers = {}

        def fit_single_classifier(clf):
            """Fits a single classifier and returns its name and instance."""
            start = time.time()
            if self.verbose:
                logger.info("Fitting classifier: %s", clf.__class__.__name__)
            clf.fit(self.feature_matrix, self.labels)

            duration_single = time.time() - start
            if self.verbose:
                logger.info(f"Classifier {clf.__class__.__name__} completed in {duration_single} seconds.")
            return clf.__class__.__name__, clf

        # Using ThreadPoolExecutor to fit classifiers asynchronously
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(fit_single_classifier, clf): clf for clf in self.classifiers}

            for future in as_completed(futures):
                clf_name, fitted_clf = future.result()
                self.fitted_classifiers[clf_name] = fitted_clf
                if self.verbose:
                    logger.info("Fitted classifier: %s", clf_name)

        end_time = time.time()  # End timing
        duration = end_time - start_time

        if self.verbose:
            logger.info("Fitting completed in %.2f seconds.", duration)


    def predict_with_classifiers(self, new_dataset_path, percentage=100):
        """
        Predicts the output for a new dataset using all fitted classifiers and stores the results.

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
        new_feature_matrix, new_labels = self.feature_strategy.run(new_loader.get_loader())

        # Store predictions in the class attribute
        self.predictions = {"GT": new_labels, }
        for clf_name, clf in self.fitted_classifiers.items():

            self.predictions[clf_name] = clf.predict(new_feature_matrix)
            if self.verbose:
                logger.info("Predictions made with classifier: %s", clf_name)

        return self.predictions

    def calculate_metrics(self, metrics=['accuracy', 'precision', 'recall', 'f1']):
        """
        Calculates specified metrics for each classifier's stored predictions.

        Args:
        true_labels (array-like): Array of true labels for comparison.
        metrics (list of str): List of metrics to calculate; options are 'accuracy', 'precision', 'recall', 'f1'.

        Returns:
        dict: A dictionary with classifier names as keys and calculated metrics as values.

        Raises:
        RuntimeError: If predictions are not available.
        """
        if not self.predictions:
            raise RuntimeError("No predictions available. Run 'predict_with_classifiers' first.")

        results = {}
        for clf_name, clf_predictions in self.predictions.items():
            clf_metrics = {}
            if 'accuracy' in metrics:
                clf_metrics['accuracy'] = accuracy_score(self.predictions["GT"], clf_predictions)
            if 'precision' in metrics:
                clf_metrics['precision'] = precision_score(self.predictions["GT"], clf_predictions, average='weighted')
            if 'recall' in metrics:
                clf_metrics['recall'] = recall_score(self.predictions["GT"], clf_predictions, average='weighted')
            if 'f1' in metrics:
                clf_metrics['f1'] = f1_score(self.predictions["GT"], clf_predictions, average='weighted')

            results[clf_name] = clf_metrics

            if self.verbose:
                logger.info("Metrics for classifier %s: %s", clf_name, clf_metrics)

        return results

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