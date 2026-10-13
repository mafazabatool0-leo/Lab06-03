
import pandas as pd
import matplotlib.pyplot as plt

data = {
    "Date": [
        "2026-01-01", "2026-01-02", "2026-01-03",
        "2026-01-04", "2026-01-05", "2026-01-06",
        "2026-01-07", "2026-01-08", "2026-01-09",
        "2026-01-10", "2026-01-11", "2026-01-12",
        "2026-01-13", "2026-01-14", "2026-01-15",
        "2026-01-01"
    ],
    "City": [
        " lahore", "ISLAMABAD ", "Karachi", "lahore ",
        "ISLAMABAD", "karachi ", "Lahore", "islamabad",
        " KARACHI", "Lahore ", "Islamabad", "KARACHI",
        "lahore", "ISLAMABAD ", "Karachi", " lahore"
    ],
    "Temperature": [
        "32C", "28C", "35C", "30C", "27C",
        "34C", "31C", "26C", "36C", "33C",
        "29C", "37C", "32C", "28C", "35C", "32C"
    ],
    "Humidity": [
        45, 55, None, 60, 65, 40, 50, None,
        35, 48, 58, 30, None, 62, 42, 45
    ],
    "Condition": [
        "sunny", " CLOUDY", "Rainy ", "SUNNY",
        "cloudy ", " sunny", "CLOUDY", "Rainy",
        "SUNNY ", "cloudy", "Sunny", "RAINY",
        " cloudy ", "Sunny ", "CLOUDY", "SUNNY"
    ]
}

df = pd.DataFrame(data)

df["City"] = df["City"].str.strip().str.title()
df["Condition"] = df["Condition"].str.strip().str.title()

df["Temperature"] = (
    df["Temperature"]
    .str.replace("C", "", regex=False)
)
df["Temperature"] = pd.to_numeric(df["Temperature"], errors="coerce")

df["Date"] = pd.to_datetime(df["Date"])

df = df.drop_duplicates().copy()

print("Rows after dropping duplicates:", len(df))

humidity_median = df["Humidity"].median()
df["Humidity"] = df["Humidity"].fillna(humidity_median)

print("Missing Humidity filled with median:", humidity_median)

print("\nCleaned Weather Data:")
print(df)

df_encoded = pd.get_dummies(
    df,
    columns=["Condition"],
    dtype=int
)

condition_columns = [
    column for column in df_encoded.columns
    if column.startswith("Condition_")
]

print("\nOne-Hot Encoded Conditions:")
print(df_encoded[condition_columns])

correlation = df["Temperature"].corr(df["Humidity"])
print("\nCorrelation (Temperature, Humidity):", round(correlation, 3))

condition_counts = df["Condition"].value_counts()
print("\nCondition counts:")
print(condition_counts)

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

axes[0, 0].plot(
    df["Date"],
    df["Temperature"],
    marker="o"
)
axes[0, 0].set_title("Temperature Over Time")
axes[0, 0].set_xlabel("Date")
axes[0, 0].set_ylabel("Temperature (C)")
axes[0, 0].tick_params(axis="x", rotation=30)

axes[0, 1].scatter(
    df["Temperature"],
    df["Humidity"]
)
axes[0, 1].set_title("Temperature vs Humidity")
axes[0, 1].set_xlabel("Temperature (C)")
axes[0, 1].set_ylabel("Humidity (%)")

axes[1, 0].pie(
    condition_counts.values,
    labels=condition_counts.index,
    autopct="%1.1f%%",
    startangle=90
)
axes[1, 0].set_title("Weather Condition Distribution")

axes[1, 1].hist(
    df["Temperature"],
    bins=5,
    edgecolor="black"
)
axes[1, 1].set_title("Temperature Distribution")
axes[1, 1].set_xlabel("Temperature (C)")
axes[1, 1].set_ylabel("Number of Days")

fig.suptitle("Weather Overview", fontsize=16)
plt.tight_layout()
plt.savefig("weather_dashboard.png")
plt.show()

print("\nSaved weather_dashboard.png")
