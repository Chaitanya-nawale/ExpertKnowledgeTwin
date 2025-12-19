import requests

# Replace with your cluster's reachable IP or localhost if using SSH tunnel
api_url = "http://localhost:16520/v1/completions"

# Your API key used for authentication
api_key = "chaikey"

# The JSON payload to send
data = {
    "model": "mistralai/Mistral-7B-Instruct-v0.3",
    "prompt": "Write a poem about AI",
    "max_tokens": 100
}

# HTTP headers including authorization and content type
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

response = requests.post(api_url, json=data, headers=headers)

if response.status_code == 200:
    result = response.json()
    print("Response from LLM:")
    print(result)
else:
    print(f"Request failed with status code {response.status_code}: {response.text}")
