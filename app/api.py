"""Small read-only API for daily Kitwe weather conditions."""

import json
import os
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse
from urllib.request import urlopen

import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "nasa_power_kitwe_raw.csv")


def load_weather_data():
    data = pd.read_csv(DATA_PATH, skiprows=12, na_values=-999)
    data = data.rename(
        columns={
            "MO": "MONTH",
            "DY": "DAY",
            "ALLSKY_SFC_SW_DWN": "GHI",
        }
    )
    data["date"] = pd.to_datetime(
        data[["YEAR", "MONTH", "DAY"]].rename(
            columns={"YEAR": "year", "MONTH": "month", "DAY": "day"}
        )
    ).dt.strftime("%Y-%m-%d")
    return data.set_index("date")


WEATHER_DATA = load_weather_data()
NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
NASA_POWER_PARAMETERS = "ALLSKY_SFC_SW_DWN,RH2M,T2M,CLOUD_AMT"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


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


def fetch_from_nasa_power(normalized_date):
    date_for_api = normalized_date.replace("-", "")
    query = (
        f"{NASA_POWER_URL}?parameters={NASA_POWER_PARAMETERS}"
        f"&community=RE&longitude=28.2132&latitude=-12.8024"
        f"&start={date_for_api}&end={date_for_api}&format=JSON"
    )

    try:
        with urlopen(query, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError) as error:
        raise LookupError("NASA POWER could not provide data for that date") from error

    parameters = payload.get("properties", {}).get("parameter", {})
    try:
        ghi = float(parameters["ALLSKY_SFC_SW_DWN"][date_for_api])
        humidity = float(parameters["RH2M"][date_for_api])
        temperature = float(parameters["T2M"][date_for_api])
        cloud_cover = float(parameters["CLOUD_AMT"][date_for_api])
    except (KeyError, TypeError, ValueError) as error:
        raise LookupError("NASA POWER has no weather record for that date") from error

    if any(value <= -999 for value in (ghi, humidity, temperature, cloud_cover)):
        raise LookupError("NASA POWER has no weather record for that date")

    return build_weather_response(
        normalized_date,
        temperature,
        humidity,
        cloud_cover,
        "NASA POWER API",
    )


def fetch_from_open_meteo(normalized_date):
    query = (
        f"{OPEN_METEO_URL}?latitude=-12.8024&longitude=28.2132"
        f"&start_date={normalized_date}&end_date={normalized_date}"
        "&daily=temperature_2m_mean,relative_humidity_2m_mean,"
        "cloud_cover_mean,shortwave_radiation_sum&timezone=Africa%2FLusaka"
    )

    try:
        with urlopen(query, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
        daily = payload["daily"]
        temperature = float(daily["temperature_2m_mean"][0])
        humidity = float(daily["relative_humidity_2m_mean"][0])
        cloud_cover = float(daily["cloud_cover_mean"][0])
    except (HTTPError, URLError, TimeoutError, KeyError, TypeError, ValueError) as error:
        raise LookupError("No live weather data is available for that date") from error

    return build_weather_response(
        normalized_date,
        temperature,
        humidity,
        cloud_cover,
        "Open-Meteo live weather API",
    )


def weather_for_date(requested_date):
    try:
        parsed_date = date.fromisoformat(requested_date)
    except ValueError as error:
        raise ValueError("date must use YYYY-MM-DD format") from error

    normalized_date = parsed_date.isoformat()
    if parsed_date > date.today():
        raise ValueError("Weather data is not available for a future date")

    if normalized_date in WEATHER_DATA.index:
        row = WEATHER_DATA.loc[normalized_date]
        return build_weather_response(
            normalized_date,
            float(row["T2M"]),
            float(row["RH2M"]),
            float(row["CLOUD_AMT"]),
            "Local NASA POWER dataset",
        )

    try:
        return fetch_from_nasa_power(normalized_date)
    except LookupError:
        return fetch_from_open_meteo(normalized_date)


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
            self.send_json({"status": "ok", "records": len(WEATHER_DATA)})
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