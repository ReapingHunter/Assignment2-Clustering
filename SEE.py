import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

np.random.seed(42)

num_patients = 500

simulated_data = pd.DataFrame({
  "pnr": np.random.randint(1000, 9999, size=num_patients),  # Patient IDs
  "eksd": pd.date_range(start="2020-01-01", periods=num_patients, freq="D"),  # Prescription dates
  "perday": np.random.choice([1, 2, 3], size=num_patients),  # Daily dosage
  "ATC": np.random.choice(["medA", "medB", "medC"], size=num_patients),  # Drug type
  "dur_original": np.random.randint(7, 30, size=num_patients)  # Duration of medication
})

df = pd.DataFrame(simulated_data)
def See(arg1):
  C09CA01 = df[df["ATC"] == arg1]
  