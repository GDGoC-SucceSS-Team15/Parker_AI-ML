import os
from flask_cors import CORS
from flask import Flask, request, render_template, redirect, url_for, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from waitress import serve
import io

# Flask 앱 생성 및 CORS 활성화
app = Flask(__name__)
CORS(app)  # 모든 도메인에서 API 호출 가능하도록 설정

# --- Custom DepthwiseConv2D 클래스 정의 ---
from tensorflow.keras.layers import DepthwiseConv2D as OriginalDepthwiseConv2D

class CustomDepthwiseConv2D(OriginalDepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        if 'groups' in kwargs:
            kwargs.pop('groups')
        super().__init__(*args, **kwargs)

# --- 모델 로드 ---
model_path = "model/model.h5"
if os.path.exists(model_path):
    model = load_model(model_path, custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D})
else:
    print("⚠ 모델 파일을 찾을 수 없습니다. API가 정상적으로 동작하지 않을 수 있습니다.")
    model = None

# 클래스 라벨 정의
class_labels = {
    0: "소화전",
    1: "교차로 모퉁이",
    2: "버스 정류소",
    3: "어린이 보호 구역",
    4: "흰색 실선",
    5: "황색 점선",
    6: "황색 복선",
    7: "황색 실선"
}

# 이미지 전처리 함수
def preprocess_image(image, target_size=(224, 224)):
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0  
    image_array = np.expand_dims(image_array, axis=0)  
    return image_array

# --- API 엔드포인트 ---
@app.route('/')
def index():
    """API가 정상 작동하는지 확인하는 엔드포인트"""
    return jsonify({
        "message": "Flask AI Model API is running!",
        "endpoints": {
            "predict": "/predict (POST)",
            "result": "/result/<int:class_id> (GET)"
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    """이미지를 업로드하면 AI 모델이 예측 결과를 반환하는 엔드포인트"""
    if 'file' not in request.files:
        return jsonify({"error": "파일이 없습니다."}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "파일을 선택해주세요."}), 400

    if model is None:
        return jsonify({"error": "AI 모델을 찾을 수 없습니다. 서버를 확인해주세요."}), 500

    try:
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        processed_image = preprocess_image(image)

        preds = model.predict(processed_image)
        pred_class = int(np.argmax(preds, axis=1)[0]) 
        max_prob = float(np.max(preds))  

        # 신뢰도 기준 (50% 미만이면 재업로드 요청)
        threshold = 0.5
        if max_prob < threshold:
            return jsonify({"reupload": True, "message": "이미지를 다시 업로드해주세요!"})

        return jsonify({"class_id": pred_class, "confidence": max_prob})

    except Exception as e:
        return jsonify({"error": f"예측 중 오류 발생: {str(e)}"}), 500

@app.route('/result/<int:class_id>')
def result_page(class_id):
    """예측된 class_id를 기반으로 정보를 반환하는 엔드포인트"""
    if class_id in class_labels:
        return jsonify({"class_id": class_id, "label": class_labels[class_id]})
    return jsonify({"error": "해당 class_id에 대한 정보가 없습니다."}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)  # Waitress 사용 시 `debug=False`가 필요합니다.