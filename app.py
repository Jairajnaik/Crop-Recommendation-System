from flask import Flask, render_template, request, redirect, url_for, session
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from datetime import datetime
import os
import logging

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Persistent user store using file (users.txt)
USER_FILE = "users.txt"

# Constants
DATA_PATH = "Crop_recommendation_updated_only_crops.csv"
LOG_FILE = "prediction_logs.txt"
LOG_FILE_DEBUG = "debug.log"

weather_map = {'sunny': 0, 'cloudy': 1, 'rainy': 2}

def setup_logging():
    logging.basicConfig(
        filename=LOG_FILE_DEBUG,
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        filemode='a'
    )
    logging.info("Logging initialized.")

def encode_weather(weather_str):
    return weather_map.get(weather_str.lower(), 2)

def preprocess_data(df):
    df['weather'] = df['weather'].str.lower().map(weather_map)
    return df.dropna(subset=['weather'])

def train_svm_model():
    df = pd.read_csv(DATA_PATH)
    df = preprocess_data(df)
    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'soil_health', 'weather']]
    y = df['label']
    model = SVC(kernel='rbf', probability=True)
    model.fit(X, y)
    logging.info("Model trained successfully.")
    return model

model = train_svm_model()

def validate_input(form):
    try:
        data = {
            'N': float(form['N']),
            'P': float(form['P']),
            'K': float(form['K']),
            'temperature': float(form['temperature']),
            'humidity': float(form['humidity']),
            'ph': float(form['ph']),
            'rainfall': float(form['rainfall']),
            'soil_health': float(form['soil_health']),
            'weather': form['weather']
        }
        if data['weather'].lower() not in weather_map:
            raise ValueError("Invalid weather input.")
        return data
    except Exception as e:
        logging.error(f"Validation failed: {e}")
        raise ValueError(f"Invalid input: {e}")

def prepare_features(data):
    encoded_weather = encode_weather(data['weather'])
    return np.array([[data['N'], data['P'], data['K'], data['temperature'],
                      data['humidity'], data['ph'], data['rainfall'],
                      data['soil_health'], encoded_weather]])

def log_prediction(inputs, predictions):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    entry = f"{timestamp} | Input: {inputs} | Prediction: {predictions}\n"
    with open(LOG_FILE, 'a') as f:
        f.write(entry)
    logging.info("Prediction logged.")

def get_top_predictions(features, model):
    prob_array = model.predict_proba(features)[0]
    crop_classes = model.classes_
    top_crops = sorted(zip(crop_classes, prob_array), key=lambda x: x[1], reverse=True)[:3]
    return top_crops

def save_user(username, password):
    with open(USER_FILE, "a") as f:
        f.write(f"{username}:{password}\n")

def validate_user(username, password):
    if not os.path.exists(USER_FILE):
        return False
    with open(USER_FILE, "r") as f:
        for line in f:
            stored_user, stored_pass = line.strip().split(":")
            if stored_user == username and stored_pass == password:
                return True
    return False

def user_exists(username):
    if not os.path.exists(USER_FILE):
        return False
    with open(USER_FILE, "r") as f:
        for line in f:
            stored_user, _ = line.strip().split(":")
            if stored_user == username:
                return True
    return False

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if user_exists(username):
            return render_template("register.html", error="Username already exists")
        save_user(username, password)
        return redirect(url_for("login"))
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if validate_user(username, password):
            session['logged_in'] = True
            return redirect(url_for("home"))
        return render_template("login.html", error="Invalid credentials")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop('logged_in', None)
    return redirect(url_for("login"))

@app.route("/")
def home():
    if not session.get('logged_in'):
        return redirect(url_for("login"))
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if not session.get('logged_in'):
        return redirect(url_for("login"))
    try:
        form_data = request.form
        user_input = validate_input(form_data)
        features = prepare_features(user_input)
        predictions = get_top_predictions(features, model)
        log_prediction(user_input, predictions)
        return render_template("result.html", crops=predictions)
    except ValueError as ve:
        return render_template("error.html", message=str(ve))
    except Exception as e:
        logging.error("Unexpected error: %s", e)
        return render_template("error.html", message=f"Unexpected error: {e}")

if __name__ == "__main__":
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w") as f:
            f.write("Crop Prediction Log\n=====================\n")
    setup_logging()
    app.run(debug=True)
