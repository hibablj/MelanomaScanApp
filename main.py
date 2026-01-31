import os
import uuid
import numpy as np
from flask import Flask, request, render_template, redirect, jsonify
import mysql.connector

# --- Limit TensorFlow threads before importing keras ---
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from tensorflow.keras.models import load_model
from PIL import Image

# --- Database connection ---
def get_db_connection():
    try:
        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="hiba",
            database="melanoma_db"
        )
        return connection
    except mysql.connector.Error as err:
        print(f"Database connection error: {err}")
        return None

# --- Flask app ---
app = Flask(__name__)

# --- Load pre-trained model ---
print("🔹 Loading model...")
try:
    model = load_model("melanoma_model.h5")
    print("✅ Model loaded")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

image_size = (128, 128)

# --- Prediction function ---
def predict_image(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
        image = image.resize(image_size)
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)
        prediction = model.predict(image)
        return "Malignant" if prediction[0][0] > 0.5 else "Benign"
    except Exception as e:
        print(f"Prediction error: {e}")
        return "Error"

# --- Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/test')
def test_melanoma():
    return render_template('test.html')

@app.route('/patients')
def list_patients():
    connection = get_db_connection()
    if not connection:
        return "Database connection failed", 500
    cursor = connection.cursor()
    specific_columns_with_alias = {
        'id': 'Identifiant',
        'name': 'Nom',
        'gender': 'Sexe',
        'date_of_birth': 'Date de naissance',
        'melanoma_test_result': 'Résultat du test de mélanome',
        'upload_date': 'Date d\'importation'
    }
    columns_string = ', '.join(f"`{col}` AS `{alias}`" for col, alias in specific_columns_with_alias.items())
    cursor.execute(f"SELECT {columns_string} FROM `patients`")
    patients = cursor.fetchall()
    columns = list(specific_columns_with_alias.values())
    cursor.close()
    connection.close()
    return render_template('patients.html', patients=patients, columns=columns)

@app.route('/predict', methods=['POST'])
def upload_file():
    name = request.form.get('name')
    gender = request.form.get('gender')
    date_naissance = request.form.get('dob')

    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    os.makedirs('uploads', exist_ok=True)
    filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
    image_path = os.path.join('uploads', filename)
    file.save(image_path)

    # Predict
    result = predict_image(image_path)

    # Insert into DB
    connection = get_db_connection()
    if connection:
        cursor = connection.cursor()
        sql = """
        INSERT INTO patients (name, gender, date_of_birth, melanoma_test_result, image_path)
        VALUES (%s, %s, %s, %s, %s)
        """
        values = (name, gender, date_naissance, result, image_path)
        cursor.execute(sql, values)
        connection.commit()
        cursor.close()
        connection.close()

    return render_template('result.html', result=result, name=name, sexe=gender, date_naissance=date_naissance)

# --- API routes ---
@app.route('/api/gender_counts')
def get_gender_counts():
    connection = get_db_connection()
    if not connection:
        return jsonify([]), 500
    cursor = connection.cursor()
    cursor.execute("SELECT gender, COUNT(*) FROM patients GROUP BY gender")
    results = cursor.fetchall()
    cursor.close()
    connection.close()
    return jsonify(results)

@app.route('/api/test_result_counts')
def get_test_result_counts():
    connection = get_db_connection()
    if not connection:
        return jsonify([]), 500
    cursor = connection.cursor()
    cursor.execute("SELECT melanoma_test_result, COUNT(*) FROM patients GROUP BY melanoma_test_result")
    results = cursor.fetchall()
    cursor.close()
    connection.close()
    test_results = [{'melanoma_test_result': row[0], 'count': row[1]} for row in results]
    return jsonify(test_results)

@app.route('/api/age_distribution')
def get_age_distribution():
    connection = get_db_connection()
    if not connection:
        return jsonify([]), 500
    cursor = connection.cursor()
    query = """
    SELECT 
        CASE 
            WHEN TIMESTAMPDIFF(YEAR, date_of_birth, CURDATE()) BETWEEN 0 AND 17 THEN '0-17'
            WHEN TIMESTAMPDIFF(YEAR, date_of_birth, CURDATE()) BETWEEN 18 AND 29 THEN '18-29'
            WHEN TIMESTAMPDIFF(YEAR, date_of_birth, CURDATE()) BETWEEN 30 AND 44 THEN '30-44'
            WHEN TIMESTAMPDIFF(YEAR, date_of_birth, CURDATE()) BETWEEN 45 AND 59 THEN '45-59'
            ELSE '60+' 
        END AS age_group,
        COUNT(*) AS count
    FROM patients
    WHERE melanoma_test_result = 'Malignant'
    GROUP BY age_group
    """
    cursor.execute(query)
    results = cursor.fetchall()
    cursor.close()
    connection.close()
    age_distribution = [{'age_group': row[0], 'count': row[1]} for row in results]
    return jsonify(age_distribution)

@app.route('/api/positive_test_counts_by_gender')
def get_positive_test_counts_by_gender():
    connection = get_db_connection()
    if not connection:
        return jsonify([]), 500
    cursor = connection.cursor()
    cursor.execute("""
        SELECT gender, COUNT(*) 
        FROM patients 
        WHERE melanoma_test_result = 'Malignant' 
        GROUP BY gender
    """)
    results = cursor.fetchall()
    cursor.close()
    connection.close()
    positive_test_counts = [{'gender': row[0], 'count': row[1]} for row in results]
    return jsonify(positive_test_counts)

# --- Run Flask ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True, use_reloader=False, threaded=False)
