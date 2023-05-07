import os , uuid
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
                id = uuid.uuid4()
                print(id)
                file_name = str(id) + '.wav'
                print(file_name)
                author = request_data['author']

        completion_status = create_audio(input, file_name)
        if completion_status == True:
            path_to_file = "./" + file_name
            upload_file(file_name, UPLOAD_BUCKET)
            url = f"https://mariabot.s3.ap-south-1.amazonaws.com/{file_name}"
            return {
                "name": file_name,
                "src": url,
                "author": author
            }

            # return send_file(
            #     path_to_file, mimetype="audio/wav", as_attachment=True, download_name=file_name)
    finally:
        os.remove(file_name)

if __name__ == "__main__":
    app.run(host='0.0.0.0')
