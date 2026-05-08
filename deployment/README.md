# Iris Species Predictor API

### What the model does
This API serves a Random Forest classifier trained on the classic Iris dataset. It accepts sepal and petal measurements (length and width in centimeters) and predicts the specific species of the Iris flower: setosa, versicolor, or virginica. The API returns both the final predicted class and the confidence probabilities for all three potential classes.

### How to run
1. Ensure you have Python installed and the required packages: `pip install flask scikit-learn joblib numpy`.
2. Ensure `model.joblib` and `target_names.joblib` are present in the same directory as the script.
3. Start the Flask server by running:
   ```bash
   python app.py

{ "features": [sepal_length, sepal_width, petal_length, petal_width] }
{
  "predicted_class": "setosa",
  "probabilities": { "setosa": 1.0, "versicolor": 0.0, "virginica": 0.0 }
}

{ "samples": [ [5.1, 3.5, 1.4, 0.2], [6.2, 3.4, 5.4, 2.3] ] } 

{
  "predictions": [
    { "predicted_class": "setosa", "probabilities": {...} },
    { "predicted_class": "virginica", "probabilities": {...} }
  ]
}

curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"features": [5.1, 3.5, 1.4, 0.2]}'