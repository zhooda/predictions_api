import flask
from flask import request, jsonify
from predict import Predict

app = flask.Flask(__name__)
app.config["DEBUG"] = False

p = Predict()

@app.route('/api/v1/predict', methods=['GET'])
def api_id():
    if 'text' in request.args and len(request.args['text']) > 0:
        text = request.args['text']
        prediction = p.get_prediction(text).astype(float)
        print(text, prediction)
        results = {
            "status": 200,
            "prediction": prediction
        }
    else:
        results = {
            "status": 400,
            "error": "no text provided"
        }
    
    return jsonify(results), 200 if results['status'] == 200 else 400

app.run(host='0.0.0.0', port=6969)