import requests
import json

url = 'http://127.0.0.1:5000/predict'
# Pointing to one of the test files from your data/test/ folder
file_path = r'C:\Users\Goutham\Downloads\assignment-1\data\test\95980236-Rental-Agreement.png' 

print(f"Sending {file_path} to API for extraction...")

try:
    with open(file_path, 'rb') as f:
        response = requests.post(url, files={'file': f})

    # Print the extracted JSON metadata
    print(json.dumps(response.json(), indent=2))
except FileNotFoundError:
    print(f"Error: Could not find {file_path}. Please check your test folder.")