from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input
import warnings
import time  

app = Flask(__name__)
CORS(app, resources={r"/upload": {"origins": "http://localhost:5173"}})

app.config['UPLOAD_FOLDER'] = 'uploads'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)

# Carregue o modelo
model = load_model("keras_model.h5", compile=False)

# Defina o formato da entrada
input_layer = Input(shape=(224, 224, 3))  # Agora estamos usando 3 canais

# Lista de classes aceitáveis
classes_aceitaveis = ["Bicicross", "Downhill", "Gravel", "Speed", "Objetos"]

# Função para processar a imagem
def process_image(image):
    try:
        image = Image.open(image).convert("RGB")
        image = image.resize((224, 224))
        color_image = np.array(image)
        
        normalized_image_array = (color_image.astype(np.float32) / 127.5) - 1

        prediction = model.predict(np.array([normalized_image_array]))    
        index = np.argmax(prediction)
        class_name = classes_aceitaveis[index]

        if class_name not in classes_aceitaveis:
            return "Classe não aceitável."
        return color_image, class_name  
    
    except Exception as e:
        return f"Erro ao processar a imagem: {str(e)}"
@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        foto_1 = request.files.get('foto1')
        foto_2 = request.files.get('foto2')
        foto_3 = request.files.get('foto3')

        if not (foto_1 and foto_2 and foto_3):
            return jsonify(error="Algum dos arquivos está faltando"), 400

        # Processar as imagens
        resulting_images = []
        for image in [foto_1, foto_2, foto_3]:
            resulting_image, class_name = process_image(image)
            if class_name not in classes_aceitaveis:
                return jsonify(error="Classe não aceitável"), 400
            resulting_images.append(resulting_image)

        # Salvar a imagem processada
        processed_image_path = os.path.join(app.config['UPLOAD_FOLDER'], "processed_image.jpg")
        cv2.imwrite(processed_image_path, np.hstack(resulting_images))

        # Preparar a resposta
        response = make_response(jsonify(result=processed_image_path, class_name=class_name))
        response.headers['Access-Control-Allow-Origin'] = 'http://localhost:5173'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        response.headers['Access-Control-Allow-Methods'] = 'OPTIONS, POST'

        return response

    except Exception as e:
        return jsonify(error=f"Erro ao processar a imagem: {str(e)}"), 500

if __name__ == '__main__':
    app.run(debug=False, port=5001)
# from flask import Flask, request, jsonify, make_response
# from flask_cors import CORS
# import os
# import cv2
# import numpy as np
# from PIL import Image
# from tensorflow.keras.models import load_model
# from tensorflow.keras.layers import Input
# import warnings

# app = Flask(__name__)
# CORS(app, resources={r"/upload": {"origins": "http://localhost:5173"}})

# app.config['UPLOAD_FOLDER'] = 'uploads'
# if not os.path.exists(app.config['UPLOAD_FOLDER']):
#     os.makedirs(app.config['UPLOAD_FOLDER'])

# with warnings.catch_warnings():
#     warnings.filterwarnings("ignore", category=DeprecationWarning)

# model = load_model("keras_model.h5", compile=False)
# input_layer = Input(shape=(224, 224, 3))  # Input shape

# classes_aceitaveis = ["Bicicross", "Downhill", "Gravel", "Speed"]

# def process_image(image):
#     try:
#         image = Image.open(image).convert("RGB")
#         image = image.resize((224, 224))
#         color_image = np.array(image)
#         normalized_image_array = (color_image.astype(np.float32) / 127.5) - 1

#         prediction = model.predict(np.array([normalized_image_array]))
#         index = np.argmax(prediction)
#         class_name = classes_aceitaveis[index]

#         if class_name not in classes_aceitaveis:
#             return "Classe não aceitável.", 0.0  # Valor de score não aceitável

#         score = float(max(prediction[0]))  # Score é a probabilidade mais alta
#         return color_image, class_name, score

#     except Exception as e:
#         return f"Erro ao processar a imagem: {str(e)}", 0.0  # Valor de score de erro

# @app.route('/upload', methods=['POST'])
# def upload_file():
#     try:
#         foto_1 = request.files.get('foto1')
#         foto_2 = request.files.get('foto2')
#         foto_3 = request.files.get('foto3')

#         if not (foto_1 and foto_2 and foto_3):
#             return jsonify(error="Algum dos arquivos está faltando"), 400

#         resulting_images = []
#         class_names = []
#         scores = []
#         for image in [foto_1, foto_2, foto_3]:
#             resulting_image, class_name, score = process_image(image)
#             if class_name not in classes_aceitaveis:
#                 return jsonify(error="Classe não aceitável"), 400
#             resulting_images.append(resulting_image)
#             class_names.append(class_name)
#             scores.append(score)

#         processed_image_path = os.path.join(app.config['UPLOAD_FOLDER'], "processed_image.jpg")
#         cv2.imwrite(processed_image_path, np.hstack(resulting_images))

#         response = make_response(jsonify(result=processed_image_path, class_names=class_names, scores=scores))
#         response.headers['Access-Control-Allow-Origin'] = 'http://localhost:5173'
#         response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
#         response.headers['Access-Control-Allow-Methods'] = 'OPTIONS, POST'

#         return response

#     except Exception as e:
#         return jsonify(error=f"Erro ao processar a imagem: {str(e)}"), 500

# if __name__ == '__main__':
#     app.run(debug=False, port=5000)
