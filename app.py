from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model and target names at startup so it's done only once
model = joblib.load("model.joblib")
target_names = joblib.load("target_names.joblib")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    # Attempt to parse JSON safely
    data = request.get_json(silent=True)
    
    if data is None or "features" not in data:
        return jsonify({"error": "Missing 'features' key in JSON body."}), 400
        
    features = data["features"]
    
    # Validate length
    if not isinstance(features, list) or len(features) != 4:
        return jsonify({"error": "The 'features' list must contain exactly 4 values."}), 400
        
    # Validate data types
    if not all(isinstance(x, (int, float)) for x in features):
        return jsonify({"error": "All feature values must be numeric."}), 400
        
    # Make prediction
    X = np.array([features])
    pred_idx = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    
    # Format probabilities mapping
    prob_dict = {target_names[i]: float(probabilities[i]) for i in range(len(target_names))}
    
    return jsonify({
        "predicted_class": target_names[pred_idx],
        "probabilities": prob_dict
    }), 200

@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    data = request.get_json(silent=True)
    
    if data is None or "samples" not in data:
        return jsonify({"error": "Missing 'samples' key in JSON body."}), 400
        
    samples = data["samples"]
    
    if not isinstance(samples, list):
        return jsonify({"error": "'samples' must be a list of feature arrays."}), 400
        
    # Basic validation for each sample in the batch
    for i, features in enumerate(samples):
        if not isinstance(features, list) or len(features) != 4:
            return jsonify({"error": f"Sample at index {i} does not have exactly 4 values."}), 400
        if not all(isinstance(x, (int, float)) for x in features):
            return jsonify({"error": f"Sample at index {i} contains non-numeric values."}), 400
            
    # Process batch prediction
    X_batch = np.array(samples)
    pred_indices = model.predict(X_batch)
    batch_probs = model.predict_proba(X_batch)
    
    results = []
    for i in range(len(samples)):
        prob_dict = {target_names[j]: float(batch_probs[i][j]) for j in range(len(target_names))}
        results.append({
            "predicted_class": target_names[pred_indices[i]],
            "probabilities": prob_dict
        })
        
    return jsonify({"predictions": results}), 200

if __name__ == "__main__":
    app.run(debug=True, port=5000)