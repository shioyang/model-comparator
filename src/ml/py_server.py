from flask import Flask, request

app = Flask(__name__)

@app.route("/")
def hello():
    return '{ "text": "Hello World!" }'


@app.route("/ml", methods=['GET', 'POST'])
def route_ml():
    file_path = ''
    print(request.method)
    print(request.form['file_path'])
    if request.method == 'POST':
        file_path = request.form['file_path']
    if file_path != '':
        return predict_file(file_path)
    else:
        return 'File path not provided.'

def predict_file(file_path):
    print('predict')
    return 'predicted: ' + file_path
