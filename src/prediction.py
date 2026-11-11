
import pandas as pd


def predict_disease(system, selected_symptoms):

    if system is None:
        raise ValueError("Model system is not loaded.")

    model = system["model"]
    features = system["feature_columns"]
    selected = set(selected_symptoms)

    # Check selected symptoms.
    invalid = selected - set(features)

    if invalid:
        raise ValueError(
            f"Invalid symptoms: {sorted(invalid)}"
        )

    # Build input using the exact training feature order.
    X = pd.DataFrame(
        [[int(feature in selected) for feature in features]],
        columns=features
    )

    # Predict disease.
    disease = model.predict(X)[0]
    probabilities = []

    # Retrieve prediction probabilities.
    if hasattr(model, "predict_proba"):

        scores = model.predict_proba(X)[0]
        classes = getattr(
            model,
            "classes_",
            system.get("classes", [])
        )

        if len(classes) != len(scores):
            raise ValueError(
                "Mismatch between model classes and probabilities."
            )

        probabilities = sorted(
            zip(classes, scores),
            key=lambda item: float(item[1]),
            reverse=True
        )

    return str(disease), probabilities
