# marimo==0.9.0
# Author email: 22f1000120@ds.study.iitm.ac.in
# This is a reactive Marimo notebook. Open it with: `marimo run marimo_notebook.py`
# It demonstrates:
#  - Variable dependencies across cells
#  - An interactive slider widget
#  - Dynamic markdown output that reacts to the widget
#  - Comments describing data flow between cells

import marimo as mo

app = mo.app()


@app.cell
def __(_mo=mo):
    # [Cell A] Setup & data source
    # Data flow: Produces an array `x` that downstream cells depend on.
    # (Downstream: Cell B uses `x` to compute `y`)
    import numpy as np

    x = np.linspace(0, 10, 500)
    return np, x


@app.cell
def __(np, x):
    # [Cell B] Transformation depending on Cell A
    # Data flow: Consumes `x` from Cell A to compute `y`.
    # (Downstream: Cell D consumes `y` and the UI state)
    y = np.sin(x)
    return y


@app.cell
def ___(_mo=mo):
    # [Cell C] UI control (independent of data)
    # Data flow: Produces a reactive slider `slider`.
    # (Downstream: Cell D consumes `slider.value` to control smoothing)
    slider = mo.ui.slider(1, 25, step=1, value=5,
                          label="Moving average window (k)")
    # Display the widget in the notebook UI:
    slider
    return slider


@app.cell
def ____(np, y, slider, _mo=mo):
    # [Cell D] Computation that depends on both data (Cell B) and UI state (Cell C)
    # Data flow:
    #   - Consumes `y` from Cell B
    #   - Consumes `slider.value` from Cell C
    #   - Produces `y_smooth` and a dynamic markdown description
    k = max(1, int(slider.value))
    kernel = np.ones(k) / k
    y_smooth = np.convolve(y, kernel, mode="same")

    # Dynamic markdown output based on widget state:
    message = (
        "✅ Fine detail preserved (small window)."
        if k <= 4
        else ("🟨 Balanced smoothing (medium window)." if k <= 12 else "🧽 Strong smoothing (large window).")
    )

    mo.md(
        f"""
        ### Sine Smoothing Demo
        - Current moving average window: **{k}**
        - {message}

        Try adjusting the slider above to see this text and the values react instantly.
        """
    )
    return k, y_smooth


@app.cell
def _____(mo, x, y, y_smooth):
    # [Cell E] Simple visualization depending on Cells A, B, and D
    # Data flow:
    #   - Consumes `x` from Cell A, `y` from Cell B, and `y_smooth` from Cell D
    #   - Produces a small table preview and an inline plot
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.DataFrame({"x": x, "y": y, "y_smooth": y_smooth})
    preview = df.head(8)

    # Show a compact preview table
    mo.ui.table(preview)

    # Plot: show raw vs smoothed values
    plt.figure()
    plt.plot(x, y, label="sin(x)")
    plt.plot(x, y_smooth, label="smoothed")
    plt.legend()
    plt.title("Raw vs Smoothed Sine Curve")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    return df


if __name__ == "__main__":
    app.run()
