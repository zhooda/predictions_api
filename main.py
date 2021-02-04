import flask
from flask import request, jsonify
import json
from predict import Predict

app = flask.Flask(__name__)
app.config["DEBUG"] = False
app.config['JSON_SORT_KEYS'] = False

p = Predict()

@app.route('/api/v1/predict', methods=['GET'])
def api_id():
    if 'text' in request.args and len(request.args['text']) > 0:
        text = request.args['text']
        prediction = p.get_prediction(text).astype(float)
        print(text, prediction)
        results = {
            "status": 200,
            "input": text,
            "prediction": prediction
        }

        if 'verbose' in request.args and request.args['verbose'] == 'true':
            if 0.4 <= prediction < 0.5:
                v = f'skewed zeeshan: {prediction}'
            elif 0.5 < prediction <= 0.6:
                v = f'skewed inara: {prediction}'
            elif prediction == 0.5:
                v = f'equal: {prediction}'
            elif prediction > 0.6:
                v = f"inara: {prediction}"
            else:
                v = f"zeeshan: {prediction}"
            results['verbose'] = v

    else:
        results = {
            "status": 400,
            "error": "no text provided"
        }

    return jsonify(results), 200 if results['status'] == 200 else 400

app.run(host='0.0.0.0', port=6969)