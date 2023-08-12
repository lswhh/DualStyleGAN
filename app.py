import os
import base64
from flask import Flask, request, redirect, render_template, send_from_directory
from werkzeug.utils import secure_filename
from cartoonize import cartoonize
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # CORS를 사용하도록 설정합니다.
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

@app.route('/cartoonize', methods=['POST'])
def handle_cartoonize():
    data = request.json
    image_data = data['image'].split(',')[1]  # 콤마 다음부터 Base64 문자열이 시작됩니다.
    img_bytes = base64.b64decode(image_data)  # 이미지 데이터를 바이트로 바꿉니다.
    img_pil = Image.open(BytesIO(img_bytes))  # PIL 이미지 포맷으로 변환합니다.

    cartoonized_img_pil = cartoonize(img_pil)  # 여기에서 cartoonize 함수를 호출합니다.

    # 결과 이미지를 다시 Base64로 인코딩해서 클라이언트에 리턴합니다.
    buffered = BytesIO()
    cartoonized_img_pil.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return jsonify(cartoonized_image='data:image/png;base64,' + img_str)

@app.route("/")
def index():
    # images = os.listdir('./images')
    # return render_template("index.html", images=images)
    return "hello DualStyleGAN"

def allowed_file(filename):
    return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/upload", methods=["GET","POST"])
def upload_file():
    if request.method=="GET":
        return render_template('upload.html')
    target = os.path.join(APP_ROOT, 'images/')
    print(target)
    if not os.path.isdir(target):
        os.mkdir(target)
    for file in request.files.getlist("file"):
        print(file)
        filename = file.filename
        destination = "/".join([target, filename])
        print(destination)
        file.save(destination)
    return render_template("uploaded.html")

@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)

def send_image_for_filter(image):
    return render_template('filter.html', image=image)

@app.route("/filters")
def filter():
    return render_template('filters.html')

@app.url_defaults
def hashed_url_for_static_file(endpoint, values):
    if 'static' == endpoint or endpoint.endswith('.static'):
        filename = values.get('filename')
        if filename:
            if '.' in endpoint:  # has higher priority
                blueprint = endpoint.rsplit('.', 1)[0]
            else:
                blueprint = request.blueprint  # can be None too
            if blueprint:
                static_folder = app.blueprints[blueprint].static_folder
            else:
                static_folder = app.static_folder
            param_name = 'h'
            while param_name in values:
                param_name = '_' + param_name
            values[param_name] = static_file_hash(os.path.join(static_folder, filename))

def static_file_hash(filename):
    return int(os.stat(filename).st_mtime)

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5000)
