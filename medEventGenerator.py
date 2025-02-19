import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set seed for reproducibility
np.random.seed(1234)

# Initialize list for synthetic data
data = []

# We'll generate data for 20 patients (pnr: 1 to 20)
num_patients = 20

for p in range(1, num_patients + 1):
    # Each patient will have between 2 and 5 prescription events
    num_events = np.random.randint(2, 6)
    # Generate a random starting date within the first 30 days of 2020
    start_date = datetime(2020, 1, 1) + timedelta(days=np.random.randint(0, 30))
    current_date = start_date
    dates = []
    # Create a series of prescription dates with random increments (1 to 100 days)
    for i in range(num_events):
        increment = np.random.randint(1, 101)
        current_date += timedelta(days=increment)
        dates.append(current_date)
    # For each event, generate the rest of the columns
    for d in dates:
        record = {
            "pnr": p,
            "eksd": d.strftime("%m/%d/%Y"),  # Format as mm/dd/YYYY
            "perday": np.random.randint(1, 4),  # Random integer between 1 and 3
            "ATC": np.random.choice(["medA", "medB"]),
            "dur_original": np.random.randint(10, 31)  # Random duration between 10 and 30
        }
        data.append(record)

# Create DataFrame
med_events = pd.DataFrame(data)
# Sort by patient and date
med_events = med_events.sort_values(by=["pnr", "eksd"])

# Save to CSV so it can be loaded in both Python and R
med_events.to_csv("med_events.csv", index=False)

print(med_events.head())
