import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 1. Create Hypothetical Dataset
# -----------------------------

data = {
    "Shipment_ID": range(1, 31),

    "Delivery_Time_Days": [
        3, 5, 4, 6, 2, 7, 5, 4, 8, 3,
        6, 5, 4, 7, 3, 6, 5, 4, 8, 2,
        5, 6, 3, 7, 4, 5, 6, 3, 7, 4
    ],

    "Shipment_Volume": [
        120, 250, 180, 300, 100, 350, 275, 160, 400, 140,
        320, 240, 190, 360, 130, 280, 210, 170, 420, 110,
        260, 310, 150, 380, 200, 230, 290, 125, 340, 185
    ],

    "Transportation_Cost": [
        1200, 2400, 1800, 3100, 1000, 3500, 2700, 1600, 4200, 1400,
        3300, 2500, 1900, 3700, 1300, 2900, 2200, 1700, 4400, 1100,
        2600, 3200, 1500, 3900, 2100, 2400, 3000, 1250, 3500, 1950
    ],

    "Distance_km": [
        100, 200, 150, 260, 80, 300, 220, 130, 350, 110,
        280, 210, 160, 310, 90, 240, 180, 140, 370, 70,
        230, 270, 120, 330, 170, 200, 250, 100, 290, 155
    ],

    "Fuel_Cost": [
        400, 800, 600, 1000, 300, 1200, 900, 500, 1400, 450,
        1100, 850, 650, 1250, 350, 950, 700, 550, 1500, 280,
        900, 1050, 400, 1350, 650, 800, 1000, 350, 1150, 620
    ],

    "Delay_Hours": [
        2, 5, 3, 7, 1, 8, 6, 2, 10, 1,
        7, 4, 3, 9, 2, 6, 5, 3, 11, 1,
        4, 7, 2, 8, 3, 5, 6, 2, 9, 4
    ],

    "Delivery_Status": [
        "On Time", "Delayed", "On Time", "Delayed", "On Time",
        "Delayed", "Delayed", "On Time", "Delayed", "On Time",
        "Delayed", "On Time", "On Time", "Delayed", "On Time",
        "Delayed", "Delayed", "On Time", "Delayed", "On Time",
        "On Time", "Delayed", "On Time", "Delayed", "On Time",
        "Delayed", "Delayed", "On Time", "Delayed", "On Time"
    ]
}

df = pd.DataFrame(data)

# Display first few records
print("First Five Records:")
print(df.head())

# -----------------------------
# 2. Dataset Information
# -----------------------------

print("\nDataset Information:")
print(df.info())

# -----------------------------
# 3. Check Missing Values
# -----------------------------

print("\nMissing Values:")
print(df.isnull().sum())

# -----------------------------
# 4. Descriptive Statistics
# -----------------------------

print("\nDescriptive Statistics:")
print(df.describe())

# -----------------------------
# 5. Central Tendency
# -----------------------------

print("\nMean Values:")
print(df.mean(numeric_only=True))

print("\nMedian Values:")
print(df.median(numeric_only=True))

print("\nMode Values:")
print(df.mode(numeric_only=True).iloc[0])

# -----------------------------
# 6. Visualization 1
# Delivery Time Distribution
# -----------------------------

plt.figure(figsize=(8, 5))

sns.histplot(
    df["Delivery_Time_Days"],
    bins=8,
    kde=True
)

plt.title("Delivery Time Distribution")
plt.xlabel("Delivery Time (Days)")
plt.ylabel("Number of Shipments")
plt.tight_layout()
plt.show()

# -----------------------------
# 7. Visualization 2
# Transportation Cost vs Distance
# -----------------------------

plt.figure(figsize=(8, 5))

sns.scatterplot(
    x="Distance_km",
    y="Transportation_Cost",
    data=df
)

plt.title("Transportation Cost vs Distance")
plt.xlabel("Distance (km)")
plt.ylabel("Transportation Cost")
plt.tight_layout()
plt.show()

# -----------------------------
# 8. Visualization 3
# Shipment Volume vs Transportation Cost
# -----------------------------

plt.figure(figsize=(8, 5))

sns.scatterplot(
    x="Shipment_Volume",
    y="Transportation_Cost",
    data=df
)

plt.title("Shipment Volume vs Transportation Cost")
plt.xlabel("Shipment Volume")
plt.ylabel("Transportation Cost")
plt.tight_layout()
plt.show()

# -----------------------------
# 9. Visualization 4
# Delivery Status Count
# -----------------------------

plt.figure(figsize=(7, 5))

sns.countplot(
    x="Delivery_Status",
    data=df
)

plt.title("Delivery Status Count")
plt.xlabel("Delivery Status")
plt.ylabel("Number of Shipments")
plt.tight_layout()
plt.show()

# -----------------------------
# 10. Visualization 5
# Correlation Heatmap
# -----------------------------

plt.figure(figsize=(10, 7))

correlation = df.select_dtypes(
    include=np.number
).corr()

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# -----------------------------
# 11. Additional Analysis
# -----------------------------

print("\nAverage Delivery Time:")
print(df["Delivery_Time_Days"].mean())

print("\nAverage Transportation Cost:")
print(df["Transportation_Cost"].mean())

print("\nAverage Delay Hours:")
print(df["Delay_Hours"].mean())

print("\nTotal Transportation Cost:")
print(df["Transportation_Cost"].sum())

print("\nDelivery Status Summary:")
print(df["Delivery_Status"].value_counts())

# -----------------------------
# 12. Save Dataset as CSV
# -----------------------------

df.to_csv("logistics_dataset.csv", index=False)

print("\nAnalysis completed successfully.")
print("Dataset saved as logistics_dataset.csv")