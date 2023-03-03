from flask import Flask, send_file, request
from main import create_audio
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.post("/text")
def process_text():
    request_data = request.get_json()
    input = None

    if request_data:
        if 'text' in request_data:
            input = request_data['text']

    completion_status = create_audio(input)
    if completion_status == True:
        path_to_file = "./sample.wav"
        return send_file(
            path_to_file, mimetype="audio/wav", as_attachment=True, attachment_filename="sample.wav")
