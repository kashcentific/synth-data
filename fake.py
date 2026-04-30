import pandas as pd
import numpy as np

np.random.seed(42)

n = 100000  # rows (~small < 50MB)

df = pd.DataFrame({
    "user_id": np.arange(n),
    "age": np.random.randint(18, 70, n),
    "session_time_min": np.random.exponential(10, n).round(2),
    "pages_viewed": np.random.poisson(5, n),
    "clicks": np.random.poisson(3, n),
    "purchase_amount": np.random.gamma(2, 20, n).round(2),
    "device_score": np.random.rand(n).round(3),
    "location_lat": np.random.uniform(-90, 90, n),
    "location_lon": np.random.uniform(-180, 180, n),
    "is_returning_user": np.random.randint(0, 2, n)
})

df.to_csv("synthetic_user_behavior.csv", index=False)

print("Dataset created successfully!")
print(df.head())