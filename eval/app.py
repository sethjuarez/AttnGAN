import os
import time
import random
from flask import Flask, jsonify, request, abort, render_template
from applicationinsights import TelemetryClient
from applicationinsights.requests import WSGIApplication
from applicationinsights.exceptions import enable
from datetime import datetime
from miscc.config import cfg
from generator import *
from waitress import serve
from saveable import *
from miscc.profile import log
#from werkzeug.contrib.profiler import ProfilerMiddleware

enable(os.environ["TELEMETRY"])
app = Flask(__name__)
app.wsgi_app = WSGIApplication(os.environ["TELEMETRY"], app.wsgi_app)

# global generator and telemetry client
global birdmaker, tc, profile, version

def handle_exception(caption):
    if profile:
        print(sys.exc_info())
    tc.track_exception(*sys.exc_info(), properties={ 'caption': caption })
    tc.flush()
    sys.exc_clear()

@app.route('/api/v1.0/bird', methods=['POST'])
def create_bird():
    if not request.json or not 'caption' in request.json:
        abort(400)

    caption = request.json['caption']

    try:
        t0 = time.time()
        urls = birdmaker.generate(caption)
        t1 = time.time()

        response = {
            'small': urls[0],
            'medium': urls[1],
            'large': urls[2],
            'map1': urls[3],
            'map2': urls[4],
            'caption': caption,
            'elapsed': t1 - t0
        }

        if profile:
            log('/api/v1.0/bird [POST]', **response)

        return jsonify({'bird': response}), 201
    except:
        handle_exception(caption)
        abort(500)

@app.route('/api/v1.0/birds', methods=['POST'])
def create_birds():
    if not request.json or not 'caption' in request.json:
        abort(400)

    caption = request.json['caption']

    try:
        t0 = time.time()
        urls = birdmaker.generate(caption, copies=6)
        t1 = time.time()

        response = {
            'bird1' : { 'small': urls[0], 'medium': urls[1], 'large': urls[2] },
            'bird2' : { 'small': urls[3], 'medium': urls[4], 'large': urls[5] },
            'bird3' : { 'small': urls[6], 'medium': urls[7], 'large': urls[8] },
            'bird4' : { 'small': urls[9], 'medium': urls[10], 'large': urls[11] },
            'bird5' : { 'small': urls[12], 'medium': urls[13], 'large': urls[14] },
            'bird6' : { 'small': urls[15], 'medium': urls[16], 'large': urls[17] },
            'caption': caption,
            'elapsed': t1 - t0
        }

        if profile:
            log('/api/v1.0/birds [POST]', caption=caption, elapsed=response['elapsed'])
            for i in range(1, 7):
                b = 'bird{}'.format(i)
                log(b, **response[b])

        return jsonify({'bird': response}), 201
    except:
        handle_exception(caption)
        abort(500)

@app.route('/health', methods=['GET'])
def get_bird():
    if request.args.get('caption') != None:
        caption = request.args.get('caption')
    else:
        caption = 'This bird has wings that are blue and has a red belly'

    try:
        t0 = time.time()
        urls = birdmaker.generate(caption)
        t1 = time.time()

        response = {
            'small': urls[0],
            'medium': urls[1],
            'large': urls[2],
            'map1': urls[3],
            'map2': urls[4],
            'caption': caption,
            'elapsed': t1 - t0,
            'version': 'Version 2'
        }

        if profile:
            log('/health [GET]', **response)

        return render_template('main.html', **response)

    except:
        handle_exception(caption)
        abort(500)
     
@app.route('/', methods=['GET'])
def get_main():
    if profile:
        log('/ [GET]', version=version)
    return version

if __name__ == '__main__':
    # initialize global objects
    global birdmaker, tc, profile, version

    version = os.environ['VERSION'].strip('"') if 'VERSION' in os.environ else 'Standard'
    profile = 'PROFILE' in os.environ and os.environ['PROFILE'].strip('"').lower() == 'true'

    if profile:
        log('current environment vars', **os.environ)

    t0 = time.time()
    tc = TelemetryClient(os.environ["TELEMETRY"].strip('"'))

    # initialize saver
    account_name = os.environ['BLOB_ACCOUNT_NAME'].strip('"')
    account_key= os.environ['BLOB_KEY'].strip('"')
    container_name = os.environ['BLOB_CONTAINER_NAME'].strip('"')

    blob_saver = BlobSaveable(account_name, account_key, container_name)
    birdmaker = Generator('data/captions.pickle', blob_saver)
    
    # gpu based
    cfg.CUDA = os.environ['GPU'].lower() == 'true'
    tc.track_event('container initializing', {"CUDA": str(cfg.CUDA)})

    seed = datetime.now().microsecond
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(seed)

    t1 = time.time()
    tc.track_event('container start', {"starttime": str(t1-t0)})
    serve(app)
