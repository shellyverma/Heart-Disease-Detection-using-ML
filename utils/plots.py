import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    auc
)


def generate_roc_curve(
    model,
    X_test,
    y_test
):

    y_probs = model.predict_proba(
        X_test
    )[:, 1]

    fpr, tpr, thresholds = roc_curve(
        y_test,
        y_probs
    )

    roc_auc = auc(
        fpr,
        tpr
    )

    fig, ax = plt.subplots()

    ax.plot(
        fpr,
        tpr,
        label=f"AUC = {roc_auc:.2f}"
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    ax.set_xlabel(
        "False Positive Rate"
    )

    ax.set_ylabel(
        "True Positive Rate"
    )

    ax.set_title(
        "ROC Curve"
    )

    ax.legend()

    return fig
