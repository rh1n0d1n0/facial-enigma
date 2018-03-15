import os

from flask import Flask, jsonify

app = Flask(__name__)
app.config.from_object('config')

API_VERSION = '0.1'
ID_STR = 'Facial-Enigma-WebAPI'

@app.route('/api')
def index():
    return jsonify({'api': ID_STR,
                    'version': API_VERSION,
                    'result': 'success',
                   })

@app.route('/api/identify')
def indentify():
    return jsonify({'message': 'endpoint not yet implemented',
                    'result': 'error',
                   })

@app.route('/api/create-subject')
def create_subject():
    return jsonify({'message': 'endpoint not yet implemented',
                    'result': 'error',
                   })

if __name__ == '__main__':
    app.run()
