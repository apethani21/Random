import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from mlxtend.plotting import plot_decision_regions


def evaluate_and_augment_test(X_test, y_test, model):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    y_pred_prob = model.predict_proba(X_test)
    confidence = y_pred_prob.max(axis=1)
    X_test_augmented = X_test.copy()
    X_test_augmented["pred"] = y_pred
    X_test_augmented["confidence"] = confidence
    return X_test_augmented


def plot_prediction(X_test, y_test, model):
    assert "confidence" in X_test
    point_size = 9
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    ax1.scatter(X_test["x"], X_test["y"], s=point_size,
                color=X_test["pred"].map({1: "#0A0AFF", 0: "#AF0000"}))
    ax1.grid()
    ax1.set_title("Prediction")
    cmap = mpl.cm.viridis
    norm = mpl.colors.Normalize(vmin=0.5, vmax=1)
    scatter = ax2.scatter(X_test["x"], X_test["y"], c=X_test["confidence"],
                          s=point_size, cmap=cmap)
    cbar_ax = fig.add_axes([0.95, 0.1, 0.05, 0.75])
    cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                        ax=ax2, cax=cbar_ax)
    cbar.set_ticks(np.linspace(0.5, 1, 11))
    ax2.grid()
    ax2.set_title("Prediction probability")
    plt.show()
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    feat1 = "feat1" if "feat1" in X_test.columns else "x"
    feat2 = "feat2" if "feat2" in X_test.columns else "y"
    plot_decision_regions(
        X_test[[feat1, feat2]].to_numpy(), y_test.to_numpy(),
        clf=model, legend=2, ax=ax, 
        colors="#AF0000,#0A0AFF",
        scatter_kwargs={"s": 15, "alpha": 0.25}
    )
    ax.grid()
    ax.set_title("Decision region")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    plt.show()
