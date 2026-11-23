from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
from pathlib import Path

app = Flask(__name__)
CORS(app)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "crop_model.pkl"

model = joblib.load(MODEL_PATH)

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "AgroGuide ML API is running"
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        required_fields = [
            "N", "P", "K", "temperature",
            "humidity", "ph", "rainfall"
        ]

        if not data:
            return jsonify({
                "error": "JSON data is required"
            }), 400

        for field in required_fields:
            if field not in data:
                return jsonify({
                    "error": f"{field} is required"
                }), 400

        features = [[
            float(data["N"]),
            float(data["P"]),
            float(data["K"]),
            float(data["temperature"]),
            float(data["humidity"]),
            float(data["ph"]),
            float(data["rainfall"])
        ]]

        crop = model.predict(features)[0]

        return jsonify({
            "success": True,
            "crop": crop
        })

    except ValueError:
        return jsonify({
            "error": "All values must be numeric"
        }), 400

    except Exception as error:
        return jsonify({
            "error": str(error)
        }), 500

if __name__ == "__main__":
    app.run(port=5000, debug=True)