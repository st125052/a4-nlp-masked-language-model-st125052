import requests

def predict_nli_class(premise, hypothesis):
    # In Production, NEVER use hard-coded URLs. Use environment variables or even better key vaults instead.
    # This is just for demonstration purposes.
    URL = "https://swaraj.ngrok.app/predict"
    params = {
        'premise': premise,
        'hypothesis': hypothesis
    }
    return requests.get(URL, params=params).json()['predicted_class']