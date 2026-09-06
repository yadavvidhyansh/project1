

def get_location():
    try:
        response = requests.get(
            "https://ipinfo.io/json",
            timeout=5
        )

        data = response.json()

        city = data.get("city", "Unknown")
        region = data.get("region", "")
        country = data.get("country", "")

        return f"{city}, {region}, {country}"

    except Exception as e:
        print("LOCATION ERROR:", e)
        return "Sorry, I could not determine your location."


def get_current_date():
    now = datetime.datetime.now()
    return now.strftime("%A, %d %B %Y")


def get_current_time():
    now = datetime.datetime.now()
    return now.strftime("%I:%M %p")