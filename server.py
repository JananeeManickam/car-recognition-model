# from flask import Flask, request, jsonify, send_from_directory
# from flask_cors import CORS
# import subprocess, tempfile, os

# app = Flask(__name__, static_folder="dashboard", static_url_path="")
# CORS(app)  # allow local HTML access

# # Explicitly point to venv Python
# PYTHON_PATH = r"D:/PSG Tech/SEMESTER 9/car-recognition-model/venv/Scripts/python.exe"
# MAIN_SCRIPT = r"D:/PSG Tech/SEMESTER 9/car-recognition-model/main.py"

# @app.route("/")
# def serve_dashboard():
#     return app.send_static_file("index.html")


# @app.route("/predict", methods=["POST"])
# def predict():
#     if "file" not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400

#     img = request.files["file"]
#     save_path = os.path.join("results", img.filename)
#     os.makedirs("results", exist_ok=True)
#     img.save(save_path)

#     try:
#         # Run prediction using the correct Python
#         result = subprocess.run(
#             [PYTHON_PATH, MAIN_SCRIPT, "-i", save_path, "-m", "both"],
#             capture_output=True,
#             text=True
#         )
#         output = result.stdout + result.stderr
#         return jsonify({"output": output})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == "__main__":
#     app.run(debug=True, port=5000)

from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import subprocess, tempfile, os
app = Flask(__name__, static_folder="dashboard", static_url_path="")
CORS(app)  # allow local HTML access

# Explicitly point to venv Python
PYTHON_PATH = r"D:/PSG Tech/SEMESTER 9/car-recognition-model/venv/Scripts/python.exe"
MAIN_SCRIPT = r"D:/PSG Tech/SEMESTER 9/car-recognition-model/main.py"
RESULTS_DIR = r"D:/PSG Tech/SEMESTER 9/car-recognition-model/results"

@app.route("/")
def serve_dashboard():
    return app.send_static_file("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    img = request.files["file"]
    save_path = os.path.join("results", img.filename)
    os.makedirs("results", exist_ok=True)
    img.save(save_path)
    try:
        # Run prediction using the correct Python
        result = subprocess.run(
            [PYTHON_PATH, MAIN_SCRIPT, "-i", save_path, "-m", "both"],
            capture_output=True,
            text=True
        )
        output = result.stdout + result.stderr
        return jsonify({"output": output})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Serve graph images from results folder
@app.route("/results/<path:filename>")
def serve_results(filename):
    try:
        return send_file(os.path.join(RESULTS_DIR, filename))
    except Exception as e:
        return jsonify({"error": str(e)}), 404

if __name__ == "__main__":
    app.run(debug=True, port=5000)