import pandas as pd

df = pd.read_csv("crop_scheduling_10K.csv")  # Replace with your actual path
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# 🔁 Load dataset
df = pd.read_csv("crop_scheduling_10K.csv")  # Update with your dataset filename

# 🏷️ Encode and save each categorical column
categorical_cols = ['Soil Type', 'Prev Crop', 'Next Crop']
encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le
    joblib.dump(le, f"{col.replace(' ', '_')}_encoder.pkl")  # e.g. Soil_Type_encoder.pkl

import numpy as np
import joblib
from tensorflow.keras.models import load_model

# ----------------------------
# 🧠 Load Everything
# ----------------------------

model = load_model("models/crop_rotation_cnn_model.h5")

scaler = joblib.load("feature_scaler.pkl")
soil_encoder = joblib.load("Soil_Type_encoder.pkl")
prev_crop_encoder = joblib.load("Prev_Crop_encoder.pkl")
next_crop_encoder = joblib.load("Next_Crop_encoder.pkl")

# ----------------------------
# 📥 Get Input from User
# ----------------------------
def get_user_input():
    print("🌾 Enter the following details to predict the recommended next crop:\n")

    rainfall = float(input("Rainfall (mm): "))
    temp = float(input("Temperature (°C): "))
    humidity = float(input("Humidity (%): "))
    sunlight = float(input("Sunlight (hours/day): "))

    # Show available options
    soil_types = soil_encoder.classes_
    print(f"Available Soil Types: {', '.join(soil_types)}")
    soil = input("Soil Type: ").strip()
    if soil not in soil_types:
        raise ValueError(f"Invalid soil type. Choose from: {', '.join(soil_types)}")
    soil_encoded = soil_encoder.transform([soil])[0]

    fertility = float(input("Fertility (0.3 to 0.9): "))
    water = int(input("Water Availability (1 = Low, 2 = Medium, 3 = High): "))
    duration = int(input("Crop Duration (days): "))

    prev_crops = prev_crop_encoder.classes_
    print(f"Available Previous Crops: {', '.join(prev_crops)}")
    prev_crop = input("Previous Crop: ").strip()
    if prev_crop not in prev_crops:
        raise ValueError(f"Invalid previous crop. Choose from: {', '.join(prev_crops)}")
    prev_crop_encoded = prev_crop_encoder.transform([prev_crop])[0]

    features = [
        rainfall, temp, humidity, sunlight, soil_encoded,
        fertility, water, duration, prev_crop_encoded
    ]
    return np.array(features).reshape(1, -1)

# ----------------------------
# 🔮 Predict Next Crop
# ----------------------------
def predict_next_crop_from_input():
    input_data = get_user_input()
    input_scaled = scaler.transform(input_data)
    input_reshaped = input_scaled.reshape((1, input_scaled.shape[1], 1))

    prediction = model.predict(input_reshaped)
    predicted_class = np.argmax(prediction)
    predicted_crop = next_crop_encoder.inverse_transform([predicted_class])[0]

    print(f"\n✅ Recommended Next Crop: {predicted_crop}")

# ----------------------------
# 🚀 Run
# ----------------------------
if __name__ == "__main__":
    predict_next_crop_from_input()

#File to Encode & Save Label Encoders
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("crop_scheduling_10K.csv")
categorical_cols = ['Soil Type', 'Prev Crop', 'Next Crop']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    joblib.dump(le, f"{col.replace(' ', '_')}_encoder.pkl")

#Prediction File
model = load_model("crop_rotation_cnn_model.h5")
scaler = joblib.load("feature_scaler.pkl")
soil_encoder = joblib.load("Soil_Type_encoder.pkl")
prev_crop_encoder = joblib.load("Prev_Crop_encoder.pkl")
next_crop_encoder = joblib.load("Next_Crop_encoder.pkl")
