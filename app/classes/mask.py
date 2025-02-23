import requests

def predict_nli_class(premise, hypothesis):
    # In Production, NEVER use hard-coded URLs. Use environment variables or even better key vaults instead.
    # This is just for demonstration purposes.
    URL = "https://swaraj.ngrok.app/predictForA4"
    params = {
        'premise': premise,
        'hypothesis': hypothesis
    }
    result = requests.get(URL, params=params).json()