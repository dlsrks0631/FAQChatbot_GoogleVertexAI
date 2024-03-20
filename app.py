from flask import Flask, render_template, request
from data import input_data

import vertexai
from vertexai.language_models import TextGenerationModel
from google.oauth2.service_account import Credentials

key_path = 'key_path/phrasal-aegis-412103-3acc76a966ab.json'
credentials = Credentials.from_service_account_file(key_path, scopes=['https://www.googleapis.com/auth/cloud-platform'])

vertexai.init(project="phrasal-aegis-412103", location="asia-northeast3", credentials=credentials)
parameters = {
    "candidate_count": 1,
    "max_output_tokens": 512,
    "temperature": 0.2,
    "top_p": 0.8,
    "top_k": 1
}
model = TextGenerationModel.from_pretrained("text-bison-32k@002")

def setQuestion(question):
    add = f"Q: {question} A:"
    return input_data + add

app = Flask(__name__)


# Define the default route to return the index.html file
@app.route("/")
def index():
    return render_template("index.html")


# Define the /api route to handle POST requests
@app.route("/api", methods=["POST"])
def api():
    # Get the message from the POST request
    question = request.json.get("message")
    
    # message = '로그인이 차단되었습니다. 어떻게 해야할까요?'
    response = model.predict(setQuestion(question), **parameters)
    answer = response.text
    print(f"Response from Model: {answer}")

    if response is not None:
        return {'content': answer}
    else:
        return 'Failed to Generate response!'


if __name__ == '__main__':
    app.run(port=1234, debug=True, host="0.0.0.0")
