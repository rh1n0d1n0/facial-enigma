
from flask import Flask, request, jsonify
from PIL import Image

from CaptureDevices import HttpStream as http
from CaptureDevices import LocalCapture as local
from CaptureDevices import RtspStream as rtsp

app = Flask(__name__)
devs = {'http': http, 'local': local, 'rtsp': rtsp}
captures = {}

@app.route('/open-device', methods=['GET'])
def open_device():
    data = request.get_json()
    device = data['device'] if 'device' in data.keys() else None
    address = data['address'] if 'address' in data.keys() else None
    if device:
        cap = devs[device](address)
        cap.start()
        captures[str(cap)] = cap
        return jsonify({'status': 'success',
                        'message': 'capture started successfully',
                        'capture': str(cap)})

    return jsonify({'status': 'error', 'message': 'unable to start capture'})


@app.route('/stop-device')
def stop_device():
    pass

@app.route('/get-img')
def get_img():
    data = request.get_json()
    device = data['device'] if 'device' in data.keys() else None
    if device:
        cap = captures[device]
        m = cap.read()
        img = matrix_to_img(m)
        return jsonify({'img':img})

def matrix_to_img(mat):
    return Image.fromarray(mat)

if __name__ == '__main__':
    app.run()
    for thread in captures.values():
        thread.stop()
        thread.join()

