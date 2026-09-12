"""Small read-only API for daily Kitwe weather conditions."""

import json
import os
from datetime import date, datetime, time, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import urlopen


OPENWEATHER_CURRENT_URL = "https://api.openweathermap.org/data/2.5/weather"
OPENWEATHER_HISTORY_URL = "https://api.openweathermap.org/data/3.0/onecall/timemachine"
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
KITWE_LATITUDE = -12.8024
KITWE_LONGITUDE = 28.2132


def classify_temperature(value):
    if value < 18:
        return "Cool"
    if value < 25:
        return "Mild"
    return "Hot"


def classify_humidity(value):
    if value < 40:
        return "Dry"
    if value < 70:
        return "Comfortable"
    return "Humid"


def classify_cloud_cover(value):
    if value < 20:
        return "Clear"
    if value < 70:
        return "Partly cloudy"
    return "Cloudy"


def build_weather_response(normalized_date, temperature, humidity, cloud_cover, source):
    return {
        "date": normalized_date,
        "location": "Kitwe, Zambia",
        "data_source": source,
        "variables": {
            "temperature": {
                "value": round(temperature, 2),
                "unit": "°C",
                "source_variable": "T2M",
                "condition": classify_temperature(temperature),
            },
            "relative_humidity": {
                "value": round(humidity, 2),
                "unit": "%",
                "source_variable": "RH2M",
                "condition": classify_humidity(humidity),
            },
            "cloud_cover": {
                "value": round(cloud_cover, 2),
                "unit": "%",
                "source_variable": "CLOUD_AMT",
                "condition": classify_cloud_cover(cloud_cover),
            },
        },
    }


def openweather_request(url, query):
    if not OPENWEATHER_API_KEY:
        raise LookupError(
            "Set OPENWEATHER_API_KEY before requesting OpenWeather conditions"
        )

    query["appid"] = OPENWEATHER_API_KEY
    query["units"] = "metric"
    try:
        with urlopen(f"{url}?{urlencode(query)}", timeout=15) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        if error.code in (401, 403):
            raise LookupError("OpenWeather rejected the API key") from error
        if error.code == 404:
            raise LookupError("OpenWeather has no record for that date") from error
        raise LookupError("OpenWeather could not provide weather data") from error
    except (URLError, TimeoutError) as error:
        raise LookupError("OpenWeather could not be reached") from error


def fetch_from_openweather(normalized_date):
    requested_date = date.fromisoformat(normalized_date)

    if requested_date == date.today():
        payload = openweather_request(
            OPENWEATHER_CURRENT_URL,
            {"lat": KITWE_LATITUDE, "lon": KITWE_LONGITUDE},
        )
        temperature = float(payload["main"]["temp"])
        humidity = float(payload["main"]["humidity"])
        cloud_cover = float(payload["clouds"]["all"])
    else:
        timestamp = int(
            datetime.combine(requested_date, time(12), tzinfo=timezone.utc).timestamp()
        )
        payload = openweather_request(
            OPENWEATHER_HISTORY_URL,
            {"lat": KITWE_LATITUDE, "lon": KITWE_LONGITUDE, "dt": timestamp},
        )
        observations = payload.get("data", [])
        if not observations:
            raise LookupError("OpenWeather has no record for that date")
        temperature = sum(float(item["temp"]) for item in observations) / len(observations)
        humidity = sum(float(item["humidity"]) for item in observations) / len(observations)
        cloud_cover = sum(float(item["clouds"]) for item in observations) / len(observations)

    return build_weather_response(
        normalized_date,
        temperature,
        humidity,
        cloud_cover,
        "OpenWeather API",
    )


def weather_for_date(requested_date):
    try:
        parsed_date = date.fromisoformat(requested_date)
    except ValueError as error:
        raise ValueError("date must use YYYY-MM-DD format") from error

    normalized_date = parsed_date.isoformat()
    if parsed_date > date.today():
        raise ValueError("Weather data is not available for a future date")

    return fetch_from_openweather(normalized_date)


class WeatherAPIHandler(BaseHTTPRequestHandler):
    def send_json(self, payload, status_code=200):
        response = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def do_GET(self):
        request = urlparse(self.path)

        if request.path == "/api/health":
            self.send_json({"status": "ok", "provider": "OpenWeather API"})
            return

        if request.path != "/api/weather":
            self.send_json({"error": "Use /api/weather?date=YYYY-MM-DD"}, 404)
            return

        requested_date = parse_qs(request.query).get("date", [date.today().isoformat()])[0]

        try:
            self.send_json(weather_for_date(requested_date))
        except ValueError as error:
            self.send_json({"error": str(error)}, 400)
        except LookupError as error:
            self.send_json(
                {
                    "error": str(error),
                },
                502,
            )

    def log_message(self, format_string, *args):
        print(f"{self.address_string()} - {format_string % args}")


def run(host="127.0.0.1", port=8000):
    server = ThreadingHTTPServer((host, port), WeatherAPIHandler)
    print(f"Weather API running at http://{host}:{port}")
    print("Try: /api/weather or /api/weather?date=YYYY-MM-DD")
    server.serve_forever()


if __name__ == "__main__":
    run()