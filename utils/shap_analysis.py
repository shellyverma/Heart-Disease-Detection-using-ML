import shap

import matplotlib.pyplot as plt


def generate_shap_plot(

    model,
    X_test

):

    explainer = shap.TreeExplainer(
        model
    )

    shap_values = explainer.shap_values(
        X_test
    )

    fig, ax = plt.subplots()

    shap.summary_plot(
        shap_values,
        X_test,
        plot_type="bar",
        show=False
    )

    return fig