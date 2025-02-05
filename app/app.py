from flask import Flask, request, send_from_directory, jsonify
from flask_cors import CORS
import pickle

# Load the model and the scaler
with open('./models/transformer/model.pkl', 'rb') as f:
    model = pickle.load(f)

# Create the Flask app
app = Flask(__name__, static_folder='./static', static_url_path='')

# Enable CORS
CORS(app)

# Define the routes
@app.route('/')
def index_page():
    return app.send_static_file('index.html')

@app.route('/<path:path>')
def serve_custom_path(path):
    return send_from_directory('./', path)

@app.route('/predict', methods=['GET'])
def predict_price():
    input_search_text = request.args.get('search')
    prediction =  model.predict(input_search_text)
    return jsonify(prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)