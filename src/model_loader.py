
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "DMI_Predict_SVM.joblib"
TRAIN_PATH = ROOT / "data" / "train.csv"
TARGET_COLUMN = "prognosis"


def load_model():

    # 1. Check model file
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    package = joblib.load(MODEL_PATH)

    # 2. Extract model information
    if isinstance(package, dict):
        model = package.get("model")
        if model is None:
            raise ValueError(
                f"Missing model key. Available keys: "
                f"{list(package.keys())}"
            )

        package_features = package.get("feature_columns")
        classes = package.get("classes")

    else:

        model = package
        package_features = None
        classes = None

    # 3. Get feature names from the trained model
    model_features = getattr(
        model,
        "feature_names_in_",
        None
    )

    if model_features is None and hasattr(model, "named_steps"):
        final_estimator = list(
            model.named_steps.values()
        )[-1]

        model_features = getattr(
            final_estimator,
            "feature_names_in_",
            None
        )

    # 4. Get all training CSV features
    csv_features = None

    if TRAIN_PATH.exists():

        df = pd.read_csv(TRAIN_PATH)
        csv_features = [
            column
            for column in df.columns
            if column != TARGET_COLUMN
        ]

    # 5. Choose the correct feature order
    if model_features is not None:
        features = list(model_features)
        feature_source = "Trained model"

    elif package_features is not None:
        features = list(package_features)
        feature_source = "Saved model package"

    elif csv_features is not None:
        features = csv_features
        feature_source = "Training CSV"

    else:
        raise ValueError(
            "Cannot find symptom feature columns."
        )

    # 6. Validate feature count
    expected = getattr(
        model,
        "n_features_in_",
        None
    )

    if expected is not None and expected != len(features):

        raise ValueError(
            f"Feature mismatch!\n"
            f"Model expects: {expected}\n"
            f"Loaded features: {len(features)}"
        )

    # 7. Validate feature names against training data
    if csv_features is not None:

        missing = [
            feature for feature in features
            if feature not in csv_features
        ]

        if missing:
            raise ValueError(
                f"Model features missing from train.csv: "
                f"{missing[:10]}"
            )

    # 8. Get disease classes
    if classes is None:
        classes = getattr(model, "classes_", [])

    if classes is None or len(classes) == 0:
        if hasattr(model, "named_steps"):
            final_estimator = list(
                model.named_steps.values()
            )[-1]

            classes = getattr(
                final_estimator,
                "classes_",
                []
            )

    system = {
        "model": model,
        "feature_columns": features,
        "classes": list(classes),
        "feature_source": feature_source
    }

    print("\nDMI-Predict Model Loaded")
    print("-----------------------")
    print(f"Model: {type(model).__name__}")
    print(f"Total symptoms: {len(features)}")
    print(f"Feature source: {feature_source}")
    print(f"Total diseases: {len(classes)}")

    return system
