import pandas as pd
import random

rows = []

for i in range(5000):
    duration = random.randint(1, 1000)
    packet_size = random.randint(20, 1500)
    failed_logins = random.randint(0, 10)
    request_rate = random.randint(1, 500)

    # Simple rule: attack detection
    if failed_logins > 5 or request_rate > 300:
        label = 1
    else:
        label = 0

    rows.append([duration, packet_size, failed_logins, request_rate, label])

df = pd.DataFrame(rows, columns=[
    "duration",
    "packet_size",
    "failed_logins",
    "request_rate",
    "Attack_Label"
])

df.to_csv("data/custom_dataset.csv", index=False)

print("✅ Dataset created successfully!")