from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import traceback                                      # NEW: for full traceback

from src.pipeline.predict_pipeline import (
    CustomData,
    PredictPipeline,
)

application = Flask(__name__)
app = application


# ---------------------------------------------------------------------------
# Home page
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


# ---------------------------------------------------------------------------
# Predict page
# ---------------------------------------------------------------------------
@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html")

    try:
        # -------------------- gather form inputs ---------------------------
        data = CustomData(
            gender=request.form.get("gender"),
            race_ethnicity=request.form.get("ethnicity"),
            parental_level_of_education=request.form.get(
                "parental_level_of_education"
            ),
            lunch=request.form.get("lunch"),
            test_preparation_course=request.form.get("test_preparation_course"),
            # FIX: reading_score and writing_score were swapped
            reading_score=float(request.form.get("reading_score")),
            writing_score=float(request.form.get("writing_score")),
        )

        # -------------------------------------------------------------------
        # Convert to DataFrame and make prediction
        # -------------------------------------------------------------------
        pred_df = data.get_data_as_data_frame()
        print("Input DataFrame\n", pred_df)

        predict_pipeline = PredictPipeline()
        predictions = predict_pipeline.predict(pred_df)
        print("Prediction completed:", predictions)

        return render_template("home.html", results=predictions[0])

    except Exception as e:
        # Show full traceback in the terminal and return readable message
        traceback.print_exc()
        return f"Error while making prediction: {e}", 500


# ---------------------------------------------------------------------------
# Run the application (debug ON so you see full errors in browser)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
