from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello():
    return '{ "text": "Hello World!" }'


@app.route("/ml")
def route_ml():
    s1 = 'ml'
    return '{ "test": "' + s1 + '" }'
