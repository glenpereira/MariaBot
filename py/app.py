from flask import Flask, send_file
from main import create_audio

app = Flask(__name__)


@app.post("/text")
def process_text():
    completion_status = create_audio("nee oru sambhavam thanne")
    if (completion_status == True):
        path_to_file = "./sample.wav"
        return send_file(
            path_to_file, mimetype="audio/wav", as_attachment=True, attachment_filename="sample.wav")
    
