import io

from flask import Flask, request, jsonify, abort, send_file
from PIL import Image

from CaptureDevices import HttpStream as http
from CaptureDevices import LocalCapture as local
from CaptureDevices import RtspStream as rtsp

app = Flask(__name__)

# List of streaming methods
METHODS = {'http': http, 'local': local, 'rtsp': rtsp}

# JSON Response format
SUCC = {'data': None}
FAIL = {'error': {'code': None, 'message': None}}

# Used to keep track of active captures
captures = {}

@app.route('/start', methods=['POST'])
def start_capture():
    data = request.get_json()
    device = data['device'] if 'device' in data.keys() else None
    if device:
        method = METHODS[device['method']](device['address'])
        method.start()
        captures[method.id] = method
        result = {'data': {'capture': method.id}}
        return jsonify(result)

    return jsonify({'error': {'code': 500, 'message': 'unable to start capture'}})

@app.route('/stop', methods=['POST'])
def stop_device():
    data = request.get_json()
    capture = data['capture'] if 'capture' in data.keys() else None
    if capture:
        cap = captures.pop(capture)
        cap.stop()
        cap.join()
        return jsonify(SUCC)

    return jsonify({'error': {'code': 400, 'message': 'unable to stop capture'}})

@app.route('/snapshot/<capture>', methods=['GET'])
def snapshot(capture):
    cap = captures[capture] if capture in captures.keys() else None
    if cap:
        image = cap.snapshot()
        image = cv_matrix_to_png(image)
        return send_file(image, mimetype='image/png')

    abort(400)

def cv_matrix_to_png(matrix):
    image = Image.fromarray(matrix)
    byte_io = io.BytesIO()
    image.save(byte_io, 'PNG')
    byte_io.seek(0)
    return byte_io

if __name__ == '__main__':
    app.run(host='0.0.0.0')
    for thread in captures.values():
        thread.stop()
        thread.join()
