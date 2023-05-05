import os , logging
from main import create_audio
from flask import Flask, send_file, request
from flask_cors import CORS

from s3_upload import upload_file

UPLOAD_BUCKET = "mariabot"

app = Flask(__name__)
CORS(app)


@app.post("/text")
def process_text():
    try:
        request_data = request.get_json()
        input = None
        print(request_data)

        if request_data:
            if 'text' in request_data:
                input = request_data['text']
                file_name = request_data['name'] + ".wav"

        completion_status = create_audio(input, file_name)
        if completion_status == True:
            path_to_file = "./" + file_name
            upload_file(file_name, UPLOAD_BUCKET)
            return send_file(
                path_to_file, mimetype="audio/wav", as_attachment=True, download_name=file_name)
    finally:
        os.remove(file_name)

if __name__ == "__main__":
    app.run(host='0.0.0.0')
