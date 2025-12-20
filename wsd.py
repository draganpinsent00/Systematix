import requests

headers = {"Authorization": "Bearer YOUR_TOKEN"}
response = requests.get("https://api.github.com/rate_limit", headers=headers)
print(response.json())