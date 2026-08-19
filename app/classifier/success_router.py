from sklearn.pipeline import (
    Pipeline,
    FeatureUnion,
)

from sklearn.feature_extraction.text import (
    TfidfVectorizer,
)

from sklearn.preprocessing import (
    StandardScaler,
)

from sklearn.linear_model import (
    LogisticRegression,
)

from sklearn.calibration import (
    CalibratedClassifierCV,
)

from app.classifier.transformers import (
    HandcraftedFeatureTransformer,
)


QUALITY_THRESHOLD = 0.8


TIER_SCORE_COLUMNS = {
    "tier_1": "tier_1_score",
    "tier_2": "tier_2_score",
    "tier_3": "tier_3_score",
}


def build_feature_transformer():

    handcrafted_pipeline = Pipeline([

        (
            "extract",
            HandcraftedFeatureTransformer(),
        ),

        (
            "scale",
            StandardScaler(),
        ),

    ])


    return FeatureUnion([

        (
            "tfidf",

            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=2,
                max_features=40000,
                sublinear_tf=True,
            ),
        ),

        (
            "handcrafted",
            handcrafted_pipeline,
        ),

    ])


def build_success_model():

    base_model = LogisticRegression(
        max_iter=2000,
        solver="saga",
        class_weight="balanced",
        random_state=42,
    )


    return CalibratedClassifierCV(
        estimator=base_model,
        method="sigmoid",
        cv=3,
    )


def build_binary_target(
    df,
    score_column,
):

    return (
        df[score_column]
        >= QUALITY_THRESHOLD
    ).astype(int)
