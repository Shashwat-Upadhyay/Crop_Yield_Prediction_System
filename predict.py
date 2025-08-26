import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load everything
model = load_model("crop_rotation_cnn_model.h5")
scaler = joblib.load("feature_scaler.pkl")
soil_encoder = joblib.load("Soil_Type_encoder.pkl")
prev_crop_encoder = joblib.load("Prev_Crop_encoder.pkl")
next_crop_encoder = joblib.load("Next_Crop_encoder.pkl")

def get_user_input():
    print("🌾 Enter the following details:\n")
    rainfall = float(input("Rainfall (mm): "))
    temp = float(input("Temperature (°C): "))
    humidity = float(input("Humidity (%): "))
    sunlight = float(input("Sunlight (hours/day): "))

    soil_types = soil_encoder.classes_
    print(f"Available Soil Types: {', '.join(soil_types)}")
    soil = input("Soil Type: ").strip()
    if soil not in soil_types:
        raise ValueError("Invalid Soil Type")
    soil_encoded = soil_encoder.transform([soil])[0]

    fertility = float(input("Fertility (0.3 to 0.9): "))
    water = int(input("Water Availability (1-3): "))
    duration = int(input("Crop Duration (days): "))

    prev_crops = prev_crop_encoder.classes_
    print(f"Available Previous Crops: {', '.join(prev_crops)}")
    prev_crop = input("Previous Crop: ").strip()
    if prev_crop not in prev_crops:
        raise ValueError("Invalid Previous Crop")
    prev_crop_encoded = prev_crop_encoder.transform([prev_crop])[0]

    features = [
        rainfall, temp, humidity, sunlight, soil_encoded,
        fertility, water, duration, prev_crop_encoded
    ]
    return np.array(features).reshape(1, -1)

def predict_next_crop_from_input():
    input_data = get_user_input()
    input_scaled = scaler.transform(input_data)
    input_reshaped = input_scaled.reshape((1, input_scaled.shape[1], 1))
    prediction = model.predict(input_reshaped)
    predicted_class = np.argmax(prediction)
    predicted_crop = next_crop_encoder.inverse_transform([predicted_class])[0]
    print(f"\n✅ Recommended Next Crop: {predicted_crop}")

if __name__ == "__main__":
    predict_next_crop_from_input()
