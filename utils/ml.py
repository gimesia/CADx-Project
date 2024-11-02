import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, cohen_kappa_score
import logging
import pandas as pd
from sklearn.base import ClassifierMixin

from utils.feature_extraction import FeatureExtractionStrategy
from utils.loader import FactoryLoader
from utils.preprocessing import PreprocessingFactory
from sklearn.decomposition import PCA

# Set up logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLPipeline:
    def __init__(self, dataset_path, preprocessing_factory: PreprocessingFactory,
                 feature_strategy: FeatureExtractionStrategy,
                 classifiers: list[ClassifierMixin], percentage: int = 100,
                 verbose: bool = False, shuffle=False, batch_size=24):
        self.loader = FactoryLoader(path=dataset_path, factory=preprocessing_factory, percentage=percentage,
                                    batch_size=batch_size, shuffle=shuffle)
        self.feature_strategy = feature_strategy
        self.classifiers = classifiers
        self.feature_matrix = None
        self.labels = None
        self.is_extracted = False
        self.fitted_classifiers = {}
        self.predictions = {}
        self.top_features_per_classifier = {} # Dictionary to store the top 10 features for each classifier
        self.batch_size = batch_size
        self.pca = None # Control PCA application
        self.verbose = verbose  # Control logging verbosity

        if self.verbose:
            logger.info("MLPipeline initialized with dataset path: %s", dataset_path)
            logger.info("Preprocessing steps", self.loader.get_transformation_steps())

    def apply_pca_to_features(self, feature_type="lbp", n_components=20):
        """
        Apply PCA to a specified type of features while leaving other features intact.
        
        Args:
        feature_type (str): Type of features to apply PCA to (e.g., 'lbp', 'glcm', 'gabor').
        n_components (int): Number of principal components to keep.
        """
        # Separate feature names based on the specified feature type
        feature_names = self.get_feature_names()
        selected_indices = [i for i, name in enumerate(feature_names) if feature_type in name]
        non_selected_indices = [i for i in range(len(feature_names)) if i not in selected_indices]

        # Separate the selected features for PCA and the non-selected features
        selected_features = self.feature_matrix[:, selected_indices]
        non_selected_features = self.feature_matrix[:, non_selected_indices]

        # Apply PCA to the selected features
        pca = PCA(n_components=n_components)
        selected_features_pca = pca.fit_transform(selected_features)

        # Combine PCA-transformed features with non-selected features
        self.feature_matrix = np.hstack([non_selected_features, selected_features_pca])

        # Update feature names to reflect PCA-transformed components
        pca_feature_names = [f"{feature_type}_pca_{i+1}" for i in range(n_components)]
        self.feature_names = [name for i, name in enumerate(feature_names) if i in non_selected_indices] + pca_feature_names

        if self.verbose:
            logger.info(f"PCA applied to {feature_type} features. Reduced to {n_components} components.")

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

    def optional_pca(self, feature_type="lbp", n_components=5):
        if not self.is_extracted:
            raise RuntimeError("Features must be extracted before applying PCA.")

        self.apply_pca_to_features(feature_type=feature_type, n_components=n_components)

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

        for i, clf in enumerate(self.classifiers):
            start = time.time()

            classifier_key = clf.__class__.__name__ + str(i)
            if classifier_key in self.fitted_classifiers:
                continue

            if self.verbose:
                logger.info(f"Fitting classifier: {classifier_key}")

            if self.pca is not None:
                pca_matrix = self.pca.fit_transform(self.feature_matrix)
                clf.fit(pca_matrix, self.labels)
            else:
                clf.fit(self.feature_matrix, self.labels)
            self.fitted_classifiers[classifier_key] = clf

            # Check if the classifier has feature importances
            if hasattr(clf, "feature_importances_"):
                try:
                    # Get feature importances and select the top 10
                    importances = clf.feature_importances_
                    top_indices = np.argsort(importances)[-10:][::-1]  # Get indices of top 10 features
                    top_features = [(self.get_feature_names()[index], importances[index]) for index in top_indices]
                    self.top_features_per_classifier[classifier_key] = top_features

                    if self.verbose:
                        logger.info(f"Top 10 features for {classifier_key}: {top_features}")
                except Exception as e:
                    print(e)

            fit_duration = time.time() - start
            if self.verbose:
                logger.info(f"Fitted classifier: {classifier_key}; Done in {fit_duration} seconds")

        duration = time.time() - start_time
        if self.verbose:
            logger.info("Fitting completed in %.2f seconds.", duration)

    def get_top_features(self):
        """
        Returns the top 10 most relevant features for each classifier that supports feature importances.
        """
        try:
            if not hasattr(self, 'top_features_per_classifier'):
                raise RuntimeError("Top features have not been calculated. Run 'fit_classifiers' first.")

            for clf_name, top_features in self.top_features_per_classifier.items():
                print(f"\nTop 10 features for {clf_name}:")
                for feature, importance in top_features:
                    print(f"{feature}: {importance:.4f}")
        except Exception as e:
            print(e)

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
        new_loader = FactoryLoader(path=new_dataset_path, factory=self.loader.get_factory(),
                                   percentage=percentage, batch_size=self.batch_size)
        new_feature_matrix, new_labels = self.feature_strategy.run(new_loader.get_loader())

        new_feature_matrix = np.nan_to_num(new_feature_matrix) # Impute nans

        if self.pca is not None:
            new_feature_matrix = self.pca.transform(new_feature_matrix)

        # Store predictions in the class attribute
        self.predictions = {"GT": new_labels, }
        for clf_name, clf in self.fitted_classifiers.items():

            self.predictions[clf_name] = clf.predict(new_feature_matrix)
            if self.verbose:
                logger.info("Predictions made with classifier: %s", clf_name)

        return self.predictions

    def calculate_metrics(self, metrics=('accuracy', 'precision', 'recall', 'f1', 'kappa'), avg="macro"):
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
                clf_metrics['precision'] = precision_score(self.predictions["GT"], clf_predictions, average=avg)
            if 'recall' in metrics:
                clf_metrics['recall'] = recall_score(self.predictions["GT"], clf_predictions, average=avg)
            if 'f1' in metrics:
                clf_metrics['f1'] = f1_score(self.predictions["GT"], clf_predictions, average=avg)
            if 'kappa' in metrics:
                clf_metrics["kappa"] = cohen_kappa_score(self.predictions["GT"], clf_predictions)
            if 'report' in metrics:
                try:
                    report =  classification_report(self.predictions["GT"], clf_predictions)
                except Exception as e:
                    print(e)
            results[clf_name] = clf_metrics

            if self.verbose:
                logger.info("Metrics for classifier %s: %s", clf_name, clf_metrics)
                try:
                    logger.info("Classification report \n%s ", report)
                except Exception as e:
                    print(e)

        return results

    def get_feature_names(self):
        """Returns feature names extracted by the strategy."""
        return self.feature_strategy.get_feature_names()

    def get_classes(self):
        """Returns classes found in training."""
        return self.loader.get_classes()

    def get_preprocessing_steps(self):
        """Returns preprocessing steps applied in the factory"""
        return self.loader.get_transformation_steps()

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


    def load_features_from_excel(self, inp_dir="./"):
        if self.verbose:
            logger.info("Loading feature matrix from Excel...")

        step_names = "_".join(self.loader.get_transformation_steps().keys())
        if not step_names:
            step_names = 'default'