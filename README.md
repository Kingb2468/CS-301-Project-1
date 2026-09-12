# CS 301 Project

Predicting Solar Irradiance Using multiple Linear Regression: A Data-Driven Approach for Regional Management

## Weather API setup

The daily weather API uses OpenWeather. Set an API key before starting the API or dashboard:

```powershell
$env:OPENWEATHER_API_KEY = "your-openweather-api-key"
```

OpenWeather current conditions are used for today. Historical dates use the OpenWeather One Call 3.0 historical endpoint, which requires access to that endpoint on the account.
