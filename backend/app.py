from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

import tensorflow as tf
import joblib
import numpy as np
import pandas as pd
import sqlite3
import cv2
import os

from PIL import Image
from datetime import datetime

# =====================================================
# APP
# =====================================================

app = Flask(__name__)
CORS(app)

# =====================================================
# FOLDERS
# =====================================================

UPLOAD_FOLDER = "static/uploads/original_images"
HEATMAP_FOLDER = "static/uploads/heatmap_images"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(HEATMAP_FOLDER, exist_ok=True)

# =====================================================
# LOAD MODELS
# =====================================================

disease_model = tf.keras.models.load_model(
    "disease_model.h5"
)

crop_model = joblib.load(
    "crop_model.pkl"
)

fertilizer_model = joblib.load(
    "fertilizer_model.pkl"
)

# =====================================================
# LOAD CSV
# =====================================================

crop_requirements = pd.read_csv(
    "datasets/crop_requirements.csv"
)

# =====================================================
# DATABASE
# =====================================================

def init_db():

    conn = sqlite3.connect("agroai.db")

    cursor = conn.cursor()

    cursor.execute("""

        CREATE TABLE IF NOT EXISTS predictions (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            prediction_type TEXT,

            result TEXT,

            confidence REAL,

            created_at TEXT

        )

    """)

    conn.commit()
    conn.close()

init_db()

# =====================================================
# FERTILIZER LABELS
# =====================================================

fertilizer_labels = {

    0: "Urea",
    1: "DAP",
    2: "14-35-14",
    3: "28-28",
    4: "17-17-17",
    5: "20-20",
    6: "10-26-26"

}

# =====================================================
# DISEASE CLASSES
# =====================================================

class_indices = {

    'Apple___Apple_scab': 0,
    'Apple___Black_rot': 1,
    'Apple___Cedar_apple_rust': 2,
    'Apple___healthy': 3,
    'Blueberry___healthy': 4,
    'Cherry_(including_sour)___Powdery_mildew': 5,
    'Cherry_(including_sour)___healthy': 6,
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 7,
    'Corn_(maize)___Common_rust_': 8,
    'Corn_(maize)___Northern_Leaf_Blight': 9,
    'Corn_(maize)___healthy': 10,
    'Grape___Black_rot': 11,
    'Grape___Esca_(Black_Measles)': 12,
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 13,
    'Grape___healthy': 14,
    'Orange___Haunglongbing_(Citrus_greening)': 15,
    'Peach___Bacterial_spot': 16,
    'Peach___healthy': 17,
    'Pepper,_bell___Bacterial_spot': 18,
    'Pepper,_bell___healthy': 19,
    'Potato___Early_blight': 20,
    'Potato___Late_blight': 21,
    'Potato___healthy': 22,
    'Raspberry___healthy': 23,
    'Soybean___healthy': 24,
    'Squash___Powdery_mildew': 25,
    'Strawberry___Leaf_scorch': 26,
    'Strawberry___healthy': 27,
    'Tomato___Bacterial_spot': 28,
    'Tomato___Early_blight': 29,
    'Tomato___Late_blight': 30,
    'Tomato___Leaf_Mold': 31,
    'Tomato___Septoria_leaf_spot': 32,
    'Tomato___Spider_mites Two-spotted_spider_mite': 33,
    'Tomato___Target_Spot': 34,
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 35,
    'Tomato___Tomato_mosaic_virus': 36,
    'Tomato___healthy': 37

}

index_to_class = {
    v: k for k, v in class_indices.items()
}

# =====================================================
# SOLUTIONS
# =====================================================

def generate_solution(disease_name):

    disease = disease_name.lower()

    if "healthy" in disease:

        return [

            "Plant is healthy.",
            "Maintain regular irrigation.",
            "Use balanced fertilizers.",
            "Monitor crop weekly."

        ]

    elif "blight" in disease:

        return [

            "Remove infected leaves.",
            "Apply Mancozeb fungicide.",
            "Avoid overwatering.",
            "Improve airflow."

        ]

    elif "spot" in disease:

        return [

            "Remove spotted leaves.",
            "Use copper fungicide.",
            "Avoid leaf wetness.",
            "Keep field clean."

        ]

    elif "rust" in disease:

        return [

            "Apply sulfur fungicide.",
            "Remove infected parts.",
            "Use resistant varieties.",
            "Reduce humidity."

        ]

    elif "virus" in disease:

        return [

            "Remove infected plants.",
            "Control aphids.",
            "Use clean seeds.",
            "Disinfect tools."

        ]

    else:

        return [

            "Monitor crop regularly.",
            "Use general fungicide.",
            "Maintain soil nutrients.",
            "Use proper irrigation."

        ]

# =====================================================
# HOME
# =====================================================

@app.route('/')
def home():

    return render_template(
        "index.html"
    )

# =====================================================
# DISEASE PREDICTION
# =====================================================

@app.route('/predict_disease', methods=['POST'])
def predict_disease():

    try:

        file = request.files['image']

        filename = file.filename

        original_path = os.path.join(
            UPLOAD_FOLDER,
            filename
        )

        file.save(original_path)

        img = Image.open(
            original_path
        ).convert("RGB")

        img = img.resize((128,128))

        img_array = np.array(img) / 255.0

        img_array = np.expand_dims(
            img_array,
            axis=0
        )

        prediction = disease_model.predict(
            img_array
        )

        index = int(
            np.argmax(prediction)
        )

        confidence = float(
            np.max(prediction)
        ) * 100

        disease_name = index_to_class.get(
            index,
            "Unknown"
        )

        # =====================================================
        # HEATMAP
        # =====================================================

        image_cv = cv2.imread(
            original_path
        )

        heatmap = cv2.applyColorMap(
            image_cv,
            cv2.COLORMAP_JET
        )

        heatmap_path = os.path.join(
            HEATMAP_FOLDER,
            filename
        )

        cv2.imwrite(
            heatmap_path,
            heatmap
        )

        # =====================================================
        # SOLUTIONS
        # =====================================================

        solutions = generate_solution(
            disease_name
        )

        # =====================================================
        # DATABASE SAVE
        # =====================================================

        conn = sqlite3.connect(
            "agroai.db"
        )

        cursor = conn.cursor()

        cursor.execute("""

            INSERT INTO predictions (

                prediction_type,
                result,
                confidence,
                created_at

            )

            VALUES (?, ?, ?, ?)

        """, (

            "disease",
            disease_name,
            confidence,

            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )

        ))

        conn.commit()
        conn.close()

        # =====================================================
        # SEVERITY
        # =====================================================

        if confidence >= 90:

            severity = "High"

        elif confidence >= 70:

            severity = "Moderate"

        else:

            severity = "Low"

        return jsonify({

            "status":"success",

            "disease":
            disease_name,

            "confidence":
            round(confidence,2),

            "severity":
            severity,

            "solutions":
            solutions,

            "heatmap_url":

            f"/static/uploads/heatmap_images/{filename}"

        })

    except Exception as e:

        return jsonify({

            "status":"error",

            "message":str(e)

        })

# =====================================================
# CROP PREDICTION + ANALYSIS
# =====================================================

@app.route('/predict_crop', methods=['POST'])
def predict_crop():

    try:

        data = request.get_json()

        N = float(data["N"])
        P = float(data["P"])
        K = float(data["K"])

        temperature = float(
            data["temperature"]
        )

        humidity = float(
            data["humidity"]
        )

        moisture = float(
            data["moisture"]
        )

        ph = float(
            data["ph"]
        )

        rainfall = float(
            data["rainfall"]
        )

        selected_crop = data["selected_crop"]

        # =====================================================
        # AI CROP RECOMMENDATION
        # =====================================================

        crop_values = [[

            N,
            P,
            K,
            temperature,
            humidity,
            ph,
            rainfall

        ]]

        crop_prediction = crop_model.predict(
            crop_values
        )

        recommended_crop = str(
            crop_prediction[0]
        )

        # =====================================================
        # FERTILIZER MODEL
        # =====================================================

        fertilizer_values = [[

            N,
            P,
            K,
            temperature,
            humidity,
            moisture,
            ph,
            rainfall

        ]]

        fertilizer_prediction = fertilizer_model.predict(
            fertilizer_values
        )

        fertilizer_index = int(
            fertilizer_prediction[0]
        )

        fertilizer_name = fertilizer_labels.get(
            fertilizer_index,
            "Unknown"
        )

        # =====================================================
        # CSV ANALYSIS
        # =====================================================

        crop_data = crop_requirements[

            crop_requirements["Crop"]
            .str.lower()

            ==

            selected_crop.lower()

        ]

        if crop_data.empty:

            return jsonify({

                "status":"error",

                "message":"Crop not found in CSV"

            })

        required_N = int(
            crop_data.iloc[0]["N"]
        )

        required_P = int(
            crop_data.iloc[0]["P"]
        )

        required_K = int(
            crop_data.iloc[0]["K"]
        )

        # =====================================================
        # DEFICIENCY
        # =====================================================

        deficiency_N = max(
            required_N - N,
            0
        )

        deficiency_P = max(
            required_P - P,
            0
        )

        deficiency_K = max(
            required_K - K,
            0
        )

        # =====================================================
        # FERTILIZER QUANTITY
        # =====================================================

        urea_required = round(
            deficiency_N / 0.46,
            2
        )

        dap_required = round(
            deficiency_P / 0.46,
            2
        )

        mop_required = round(
            deficiency_K / 0.60,
            2
        )

        # =====================================================
        # SOIL HEALTH SCORE
        # =====================================================

        total_deficiency = (

            deficiency_N +
            deficiency_P +
            deficiency_K

        )

        health_score = 100 - (
            total_deficiency / 3
        )

        health_score = max(
            min(health_score,100),
            0
        )

        # =====================================================
        # DATABASE SAVE
        # =====================================================

        conn = sqlite3.connect(
            "agroai.db"
        )

        cursor = conn.cursor()

        cursor.execute("""

            INSERT INTO predictions (

                prediction_type,
                result,
                confidence,
                created_at

            )

            VALUES (?, ?, ?, ?)

        """, (

            "crop",
            recommended_crop,
            100,

            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )

        ))

        conn.commit()
        conn.close()

        return jsonify({

            "status":"success",

            # =====================================================
            # AI RECOMMENDATION
            # =====================================================

            "recommended_crop":
            recommended_crop,

            # =====================================================
            # ANALYSIS CROP
            # =====================================================

            "selected_crop":
            selected_crop,

            # =====================================================
            # FERTILIZER
            # =====================================================

            "fertilizer":
            fertilizer_name,

            # =====================================================
            # SOIL HEALTH
            # =====================================================

            "soil_health":
            round(
                health_score,
                2
            ),

            # =====================================================
            # CURRENT VALUES
            # =====================================================

            "current_values":{

                "N":N,
                "P":P,
                "K":K

            },

            # =====================================================
            # REQUIRED VALUES
            # =====================================================

            "required_values":{

                "N":required_N,
                "P":required_P,
                "K":required_K

            },

            # =====================================================
            # DEFICIENCY
            # =====================================================

            "deficiency":{

                "N":deficiency_N,
                "P":deficiency_P,
                "K":deficiency_K

            },

            # =====================================================
            # FERTILIZER QUANTITY
            # =====================================================

            "fertilizer_quantity":{

                "Urea":
                f"{urea_required} kg per acre",

                "DAP":
                f"{dap_required} kg per acre",

                "MOP":
                f"{mop_required} kg per acre"

            },

            # =====================================================
            # BUY LINK
            # =====================================================

            "buy_link":

            f"https://www.amazon.in/s?k={fertilizer_name}+fertilizer"

        })

    except Exception as e:

        return jsonify({

            "status":"error",

            "message":str(e)

        })

# =====================================================
# RUN
# =====================================================

if __name__ == "__main__":

    app.run(

        host="0.0.0.0",
        port=5000,
        debug=True

    )