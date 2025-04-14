from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# Load models and scaler
classifier = joblib.load('logistic_model.pkl')
svc = joblib.load('svm_model.pkl')
clf = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        features = np.array([[data['DH'], data['DL'], data['ratio'], data['repeat'],
                              data['used_chip'], data['used_pin'], data['online']]])
        features_scaled = scaler.transform(features)

        pred1 = classifier.predict(features_scaled)[0]
        pred2 = svc.predict(features_scaled)[0]
        pred3 = clf.predict(features_scaled)[0]

        total = pred1 + pred2 + pred3
        result = 1 if total >= 2 else 0  # Majority voting

        return jsonify({"fraud": result})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
