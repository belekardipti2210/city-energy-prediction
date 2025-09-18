
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)
zones = [f"Zone_{i}" for i in range(1, 6)]
dates = pd.date_range("2024-01-01", periods=365, freq="D")

rows = []
for zone in zones:
    base_usage = np.random.uniform(900, 1400)
    temp_bias = np.random.uniform(-1.5, 1.5)
    for d in dates:
        day_of_year = d.timetuple().tm_yday
        avg_temp = 15 + 10*np.sin(2*np.pi*(day_of_year-172)/365) + temp_bias + np.random.normal(0, 1.5)
        humidity = np.clip(60 - 10*np.sin(2*np.pi*(day_of_year-172)/365) + np.random.normal(0, 5), 20, 100)
        is_weekend = d.weekday() >= 5
        event_prob = 0.05 + (0.03 if is_weekend else 0.0)
        special_event = np.random.rand() < event_prob
        weekend_factor = np.random.uniform(0.90, 1.05) if is_weekend else 1.0
        temp_effect = 1.0 + 0.02 * max(0, abs(avg_temp - 22))
        hum_effect = 1.0 + 0.001 * (humidity - 50)
        event_bump = 1.15 if special_event else 1.0
        consumption = base_usage * temp_effect * hum_effect * weekend_factor * event_bump + np.random.normal(0, 50)
        rows.append({
            "Date": d,
            "ZoneID": zone,
            "AvgTemperature": round(avg_temp, 2),
            "Humidity": round(humidity, 2),
            "SpecialEvent": int(special_event),
            "EnergyConsumption": round(max(100, consumption), 2)
        })

df = pd.DataFrame(rows)
for col in ["AvgTemperature", "Humidity"]:
    mask_idx = np.random.choice(df.index, size=int(len(df)*0.005), replace=False)
    df.loc[mask_idx, col] = np.nan


df["AvgTemperature"] = df.groupby("ZoneID")["AvgTemperature"].transform(lambda x: x.fillna(x.rolling(7, min_periods=1).mean()))
df["Humidity"] = df.groupby("ZoneID")["Humidity"].transform(lambda x: x.fillna(x.rolling(7, min_periods=1).mean()))

df["AvgTemperature"].fillna(df["AvgTemperature"].mean(), inplace=True)
df["Humidity"].fillna(df["Humidity"].mean(), inplace=True)
df["Month"] = df["Date"].dt.month
df["DayOfWeek"] = df["Date"].dt.dayofweek


print("\nAverage Consumption per Zone:")
print(df.groupby("ZoneID")["EnergyConsumption"].mean())

print("\nCorrelation Matrix:")
print(df[["AvgTemperature", "Humidity", "SpecialEvent", "EnergyConsumption"]].corr())


avg_month = df.groupby("Month")["EnergyConsumption"].mean()
plt.figure(figsize=(8, 4))
plt.plot(avg_month.index, avg_month.values, marker='o')
plt.title("Monthly Avg Energy Consumption")
plt.xlabel("Month"); plt.ylabel("Consumption (kWh)")
plt.grid(True); plt.show()

corr = df[["AvgTemperature", "Humidity", "SpecialEvent", "EnergyConsumption"]].corr()
plt.imshow(corr, cmap="coolwarm", interpolation="nearest")
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
plt.yticks(range(len(corr.index)), corr.index)
plt.title("Correlation Heatmap"); plt.show()

event_avg = df.groupby("SpecialEvent")["EnergyConsumption"].mean()
plt.bar(event_avg.index.astype(str), event_avg.values)
plt.title("Avg Consumption: Event vs Non-event Days")
plt.xlabel("Special Event (0=No,1=Yes)"); plt.ylabel("Consumption (kWh)")
plt.show()


X = df[["ZoneID", "AvgTemperature", "Humidity", "SpecialEvent", "Month", "DayOfWeek"]]
y = df["EnergyConsumption"]

cat_features = ["ZoneID", "Month", "DayOfWeek"]
preprocessor = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)], remainder="passthrough")

rf_model = Pipeline([("pre", preprocessor), ("rf", RandomForestRegressor(n_estimators=200, random_state=42))])
lr_model = Pipeline([("pre", preprocessor), ("lr", LinearRegression())])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model.fit(X_train, y_train)
lr_model.fit(X_train, y_train)

rf_mae = mean_absolute_error(y_test, rf_model.predict(X_test))
lr_mae = mean_absolute_error(y_test, lr_model.predict(X_test))
print(f"\nRandomForest MAE: {rf_mae:.2f} kWh")
print(f"Linear Regression MAE: {lr_mae:.2f} kWh")

def predict_consumption():
    print("\n--- Predict Tomorrow's Energy Consumption ---")
    zone = input(f"Enter Zone ID {zones}: ").strip()
    temp = float(input("Enter tomorrow's Avg Temperature (°C): "))
    hum = float(input("Enter tomorrow's Humidity (0-100): "))
    event = int(input("Special Event? (0=No,1=Yes): "))
    month = int(input("Enter Month (1-12): "))
    dow = int(input("Enter Day of Week (0=Mon...6=Sun): "))
    row = pd.DataFrame([{
        "ZoneID": zone,
        "AvgTemperature": temp,
        "Humidity": hum,
        "SpecialEvent": event,
        "Month": month,
        "DayOfWeek": dow
    }])
    pred = rf_model.predict(row)[0]
    print(f"\nPredicted Consumption: {pred:.2f} kWh")
