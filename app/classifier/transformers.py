import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

from app.classifier.features import extract_features


class HandcraftedFeatureTransformer(
    BaseEstimator,
    TransformerMixin
):

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        feature_rows = []

        for prompt in X:

            features = extract_features(prompt)

            feature_rows.append(
                list(features.values())
            )

        return np.array(
            feature_rows,
            dtype=float
        )