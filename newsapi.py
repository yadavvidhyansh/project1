import os
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("NEWS_API_KEY")

url = "https://newsapi.org/v2/top-headlines"

params = {
    "apiKey": API_KEY,
    "country": "in",
    "pageSize": 5
}

response = requests.get(url, params=params)

print("Status Code:", response.status_code)
print("Response:")

data = response.json()
print(data)

if response.status_code == 200:
    print("\n✅ NewsAPI is working!")

    for article in data["articles"]:
        print("\nTitle:", article["title"])
        print("Source:", article["source"]["name"])

else:
    print("\n❌ NewsAPI failed!")
    print("Error:", data.get("message"))