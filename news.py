import os
import requests
from dotenv import load_dotenv

load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

url = "https://newsapi.org/v2/top-headlines"

params = {
    "q": "India",
    "apiKey": NEWS_API_KEY,
    "pageSize": 5,
    "language": "en"
}

try:
    response = requests.get(url, params=params, timeout=10)

    print("Status Code:", response.status_code)
    print(response.json())

except Exception as e:
    print("Error:", e)