import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score
)

from sklearn.preprocessing import StandardScaler


def load_and_train_models():

    df = pd.read_csv(
        "data/heart_disease_data.csv"
    )

    X = df.drop(
        "target",
        axis=1
    )

    y = df["target"]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    models = {

        "Logistic Regression":
        LogisticRegression(
            max_iter=1000
        ),

        "Random Forest":
        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),

        "Gradient Boosting":
        GradientBoostingClassifier(
            random_state=42
        ),

        "SVM":
        SVC(
            probability=True,
            random_state=42
        ),
    }

    results = {}

    trained_models = {}

    for name, model in models.items():

        model.fit(
            X_train,
            y_train
        )

        y_pred = model.predict(X_test)

        y_proba = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        cv_scores = cross_val_score(
            model,
            X_scaled,
            y,
            cv=5
        )

        report = classification_report(
            y_test,
            y_pred,
            output_dict=True
        )

        results[name] = {

            "accuracy":
            round(accuracy * 100, 2),

            "cv_mean":
            round(cv_scores.mean() * 100, 2),

            "cv_std":
            round(cv_scores.std() * 100, 2),

            "precision":
            round(
                report["1"]["precision"] * 100,
                2
            ),

            "recall":
            round(
                report["1"]["recall"] * 100,
                2
            ),

            "f1":
            round(
                report["1"]["f1-score"] * 100,
                2
            ),

            "roc_auc":
            round(
                roc_auc_score(
                    y_test,
                    y_proba
                ) * 100,
                2
            ),

            "y_pred":
            y_pred,
        }

        trained_models[name] = model

    best_model = max(
        results,
        key=lambda x: results[x]["accuracy"]
    )

    return (
        df,
        X,
        y,
        scaler,
        trained_models,
        results,
        best_model,
        X_test,
        y_test,
    )
