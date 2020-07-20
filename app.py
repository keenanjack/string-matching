from flask import Flask
app = Flask(__name__)

from pipeline import process

@app.route('/')
def hello_world():
    process()
    return 'Hello, World!'

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80, debug=True, threaded=True)