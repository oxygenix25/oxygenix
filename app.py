import numpy as np
import pandas as pd
import joblib
import os
import time
from flask import Flask, request, render_template, jsonify
import firebase_admin
from firebase_admin import credentials, db, firestore

# =============================================================================
# KONFIGURASI FIREBASE
# =============================================================================
FIREBASE_CRED_PATH = './serviceAccountKey.json'
FIREBASE_DB_URL = 'https://copd-prediction-project-11af5-default-rtdb.asia-southeast1.firebasedatabase.app/'

# Node untuk Realtime Database (Sensor)
NODE_COMMAND = 'copd_iot/command'
NODE_RESULT = 'copd_iot/result'

POLLING_TIMEOUT_SECONDS = 120
POLLING_INTERVAL_SECONDS = 1

db_firestore = None

try:
    cred = credentials.Certificate(FIREBASE_CRED_PATH)
    firebase_admin.initialize_app(cred, {'databaseURL': FIREBASE_DB_URL})
    db_firestore = firestore.client()
    print("✅ Firebase (RTDB & Firestore) berhasil diinisialisasi.")
except Exception as e:
    print(f"❌ Gagal inisialisasi Firebase: {e}")
    pass
# =============================================================================
# KONFIGURASI MODEL
# =============================================================================
MODEL_FILE = 'knn_copd_final.pkl'
CSV_FILE = 'prediction_log.csv'

try:
    with open(MODEL_FILE, 'rb') as f:
        final_model_bundle = joblib.load(f)

    loaded_model = final_model_bundle['model']
    scaler = final_model_bundle['scaler']
    le_dict = final_model_bundle['encoders']
    selected_features = final_model_bundle['selected_features']
    print(f"✅ Model '{MODEL_FILE}' berhasil dimuat.")
except Exception as e:
    print(f"❌ Error memuat model: {e}")
    exit()

# =============================================================================
# KONFIGURASI ENCODING
# =============================================================================
gender_map = {'Pria': 1, 'Wanita': 0}
activity_map = {'Tinggi': 'High', 'Sedang': 'Moderate', 'Rendah': 'Low'}

smoking_status_reverse_map = {
    'Yes': 'Perokok Aktif',
    'Former': 'Mantan Perokok',
    'No': 'Tidak Pernah Merokok'
}

RAW_FEATURES = [
    'age', 'BMI', 'heart rate', 'SP O2', 'gendera',
    'Predicted_Activity_Level', 'Predicted_Smoking_Status'
]
CATEGORICAL_COLS_TO_ENCODE = ['Predicted_Activity_Level', 'Predicted_Smoking_Status']

# =============================================================================
# INISIALISASI FLASK
# =============================================================================
app = Flask(__name__)

# =============================================================================
# HALAMAN UTAMA (INDEX.HTML)
# =============================================================================
@app.route('/')
def home():
    return render_template('index.html')

# =============================================================================
# 🔹 HALAMAN BARU (DATASET.HTML)
# =============================================================================
@app.route('/dataset')
def dataset_page():
    return render_template('dataset.html')

# =============================================================================
# GET SENSOR DATA (RTDB)
# =============================================================================
@app.route('/get_sensor_data', methods=['GET'])
def get_sensor_data():
    try:
        db.reference(NODE_RESULT).set({'HeartRate': 0, 'SPO2': 0, 'status': 'waiting'})
        db.reference(NODE_COMMAND).set('START_MEASUREMENT')
        print("🚀 Perintah START_MEASUREMENT dikirim ke Firebase RTDB.")

        start_time = time.time()
        result_data = None

        while (time.time() - start_time) < POLLING_TIMEOUT_SECONDS:
            result_node = db.reference(NODE_RESULT).get()
            if result_node and result_node.get('HeartRate', 0) > 30 and result_node.get('status') == 'success':
                print("✅ Hasil sensor diterima dari ESP32.")
                result_data = result_node
                break
            if result_node and result_node.get('status') == 'error':
                print("⚠️ ESP32 mengirim status error.")
                db.reference(NODE_COMMAND).set('STOP')
                return jsonify({'status': 'error', 'message': result_node.get('message', 'ESP32 error tidak terdefinisi.')}), 500
            time.sleep(POLLING_INTERVAL_SECONDS)

        if not result_data:
            print("⏱️ Timeout: Tidak ada respon dari ESP32.")
            db.reference(NODE_COMMAND).set('STOP')
            return jsonify({'status': 'error', 'message': f'Time-out: ESP32 tidak merespon dalam {POLLING_TIMEOUT_SECONDS} detik.'}), 504

        db.reference(NODE_COMMAND).set('STOP')
        return jsonify({'status': 'success', 'HeartRate': result_data['HeartRate'], 'SPO2': result_data['SPO2']})
    except Exception as e:
        db.reference(NODE_COMMAND).set('STOP')
        return jsonify({'status': 'error', 'message': f'Kesalahan sistem/Firebase RTDB: {str(e)}'}), 500

# =============================================================================
# PREDIKSI KNN COPD
# =============================================================================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_raw = {
            'age': float(data['age']),
            'BMI': float(data['bmi']),
            'gendera': gender_map.get(data['gender']),
            'heart rate': float(data['heart_rate']),
            'SP O2': float(data['spo2']),
            'Predicted_Activity_Level': activity_map.get(data['activity']),
            'Predicted_Smoking_Status': data['smoking_status']
        }

        sample = pd.DataFrame([input_raw], columns=RAW_FEATURES)

        sample['HR_SPO2_ratio'] = sample['heart rate'] / sample['SP O2']
        sample['BMI_age_ratio'] = sample['BMI'] / sample['age']
        sample['HRxBMI'] = sample['heart rate'] * sample['BMI']
        sample['age_BMI_ratio'] = sample['age'] / sample['BMI']
        sample['age_gender_ratio'] = sample['age'] / sample['gendera'].replace(0, np.nan)
        sample['gender_BMI_ratio'] = sample['gendera'] / sample['BMI']

        for col in CATEGORICAL_COLS_TO_ENCODE:
            if col in sample.columns and col in le_dict:
                le = le_dict[col]
                sample[col] = sample[col].map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)

        sample = sample[selected_features]
        sample_scaled = scaler.transform(sample)
        prob_copd = loaded_model.predict_proba(sample_scaled)[0][1]
        resiko_copd_persen = prob_copd * 100
        if prob_copd < 0.33:
            resiko_label = "Rendah"
        elif prob_copd < 0.66:
            resiko_label = "Sedang"
        else:
            resiko_label = "Tinggi"
        result = f"{resiko_copd_persen:.2f}% (Resiko {resiko_label})"

        # Kembalikan HANYA hasil prediksi. Data input sudah ada di frontend.
        return jsonify({'status': 'success', 'prediction_result': result})
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Kesalahan saat prediksi: {str(e)}'}), 400

# =============================================================================
# SIMPAN LOG KE CSV DAN FIRESTORE
# =============================================================================
# =============================================================================
# SIMPAN LOG KE CSV DAN FIRESTORE
# =============================================================================
@app.route('/save_csv', methods=['POST'])
def save_csv():
    try:
        # Ambil data lengkap dari request (yang sudah disiapkan frontend)
        data_log = request.get_json()
        nama_pasien = data_log.get('nama', 'Pasien_Tanpa_Nama')
        firebase_message = ""

        # -----------------------------------------------------------------
        # 🔹 TRANSLASI DATA SEBELUM DISIMPAN
        # -----------------------------------------------------------------
        try:
            # Ambil nilai English (cth: "Yes")
            english_status = data_log.get('smoking_status')
            
            # Cari padanan Indonesian-nya (cth: "Perokok Aktif")
            # .get(english_status, english_status) -> jika tidak ketemu, pakai nilai asli
            indonesian_status = smoking_status_reverse_map.get(english_status, english_status)
            
            # Timpa nilai 'smoking_status' di data_log dengan versi Bahasa Indonesia
            data_log['smoking_status'] = indonesian_status
            
        except Exception as e:
            # Jika ada error, cetak peringatan tapi jangan hentikan proses
            print(f"⚠️ Peringatan: Gagal mentranslasi smoking_status: {e}")
        # -----------------------------------------------------------------

        
        # 1. Simpan ke Firestore
        try:
            if db_firestore is None:
                raise Exception("Firestore client tidak terinisialisasi.")
            
            # data_fs sekarang disalin DARI data_log yang sudah diterjemahkan
            data_fs = data_log.copy() 
            current_time = int(time.time())
            data_fs['timestamp'] = current_time
            data_fs['timestamp_readable'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time))
            
            log_collection = 'prediction_logs'
            doc_ref = db_firestore.collection(log_collection).add(data_fs)
            
            print(f"✅ Log untuk {nama_pasien} berhasil disimpan ke Firestore (ID: {doc_ref[1].id}).")
            firebase_message = "Log Firestore juga tersimpan."
            
        except Exception as e:
            print(f"⚠️ Gagal menyimpan log ke Firestore: {str(e)}")
            firebase_message = f"Gagal simpan ke Firestore: {str(e)}"

        # 2. Simpan ke CSV
        # data_csv sekarang juga disalin DARI data_log yang sudah diterjemahkan
        data_csv = data_log.copy() 
        nama_pasien_csv = data_csv.pop('nama', 'Pasien_Tanpa_Nama')
        
        df_new_row = pd.DataFrame([data_csv])
        df_new_row.index = [nama_pasien_csv]
        df_new_row.index.name = 'Nama'

        file_exists = os.path.isfile(CSV_FILE)
        df_new_row.to_csv(CSV_FILE, mode='a', header=not file_exists, index=True)

        return jsonify({
            'status': 'success',
            'message': f'Data {nama_pasien} disimpan ke CSV. {firebase_message}'
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Gagal menyimpan data: {str(e)}'
        }), 500
# =============================================================================
# 🔹 API BARU: AMBIL SEMUA LOG DARI FIRESTORE
# =============================================================================
@app.route('/get_logs', methods=['GET'])
def get_logs():
    try:
        if db_firestore is None:
            raise Exception("Firestore client tidak terinisialisasi.")
        
        log_collection = 'prediction_logs'
        # Mengurutkan berdasarkan timestamp, dari yang terbaru
        docs = db_firestore.collection(log_collection).order_by(
            'timestamp', direction=firestore.Query.DESCENDING
        ).stream()
        
        logs = []
        for doc in docs:
            log_data = doc.to_dict()
            log_data['doc_id'] = doc.id # Tambahkan ID dokumen untuk proses hapus
            logs.append(log_data)
            
        return jsonify({'status': 'success', 'logs': logs})
    
    except Exception as e:
        print(f"⚠️ Gagal mengambil log: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

# =============================================================================
# 🔹 API BARU: HAPUS LOG DARI FIRESTORE
# =============================================================================
@app.route('/delete_log', methods=['POST'])
def delete_log():
    try:
        data = request.get_json()
        doc_id = data.get('doc_id')
        
        if not doc_id:
            return jsonify({'status': 'error', 'message': 'doc_id tidak ditemukan'}), 400

        if db_firestore is None:
            raise Exception("Firestore client tidak terinisialisasi.")

        log_collection = 'prediction_logs'
        db_firestore.collection(log_collection).document(doc_id).delete()
        
        print(f"✅ Log {doc_id} berhasil dihapus dari Firestore.")
        return jsonify({'status': 'success', 'message': f'Log {doc_id} berhasil dihapus.'})

    except Exception as e:
        print(f"⚠️ Gagal menghapus log: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')