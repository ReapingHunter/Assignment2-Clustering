import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

num_patients = 500

df = pd.DataFrame({
  "pnr": np.random.randint(1000, 1010, size=num_patients),  # Patient IDs
  "eksd": pd.date_range(start="2020-01-01", periods=num_patients, freq="D"),  # Prescription dates
  "perday": np.random.choice([1, 2, 3], size=num_patients),  # Daily dosage
  "ATC": np.random.choice(["medA", "medB", "medC"], size=num_patients),  # Drug type
  "dur_original": np.random.randint(7, 30, size=num_patients)  # Duration of medication
})


def see_kmeans(arg1):
  C09CA01 = df[df["ATC"] == arg1].copy()

  # Sorting prescriptions by patient and date
  C09CA01 = C09CA01.sort_values(by=["pnr", "eksd"])

  # Compute previous prescription date
  C09CA01["prev_eksd"] = C09CA01.groupby("pnr")["eksd"].shift(1)

  # Remove first prescriptions (as they have no previous date)
  C09CA01 = C09CA01.dropna(subset=["prev_eksd"])

  # Compute event interval (difference in days)
  C09CA01["event_interval"] = (C09CA01["eksd"] - C09CA01["prev_eksd"]).dt.days


  # Generate ECDF plot
  sns.ecdfplot(C09CA01["event_interval"])
  plt.title("Empirical Cumulative Distribution Function (ECDF)")
  plt.show

  # Keep only 80% of data by removing upper 20%
  cutoff = np.percentile(C09CA01["event_interval"], 80)
  C09CA01 = C09CA01[C09CA01["event_interval"] <= cutoff]
  
  # Density plot
  event_intervals_log = np.log(C09CA01["event_interval"])
  density = gaussian_kde(event_intervals_log)
  xs = np.linspace(event_intervals_log.min(), event_intervals_log.max(), 200)
  plt.plot(xs, density(xs))
  plt.title("Log (Event Interval) Density Plot")
  plt.show()

  # Standardize for clustering
  scaler = StandardScaler()
  event_intervals_scaled = scaler.fit_transform(event_intervals_log.values.reshape(-1, 1))

  # K-means clustering
  optimal_k = 3
  kmeans = KMeans(n_clusters=optimal_k, random_state=42)
  C09CA01["Cluster"] = kmeans.fit_predict(event_intervals_scaled)

  # Compute cluster medians
  cluster_medians = C09CA01.groupby("Cluster")["event_interval"].median().to_dict()
  C09CA01["Median"] = C09CA01["Cluster"].map(cluster_medians)

  return C09CA01

def see_assumption(df):
  df = df.sort_values(by=["pnr", "eksd"])
  df["prev_eksd"] = df.groupby("pnr")["eksd"].shift(1)
  df = df.dropna(subset=["prev_eksd"])
  df["Duration"] = (df["eksd"] - df["prev_eksd"]).dt.days
  
  # Boxplot of durations per sequence number
  df["p_number"] = df.groupby("pnr").cumcount() + 1
  plt.figure(figsize=(10, 5))
  sns.boxplot(x=df["p_number"], y=df["Duration"])
  plt.axhline(y=df["Duration"].median(), linestyle="dashed", color="red")
  plt.title("Duration Distribution Across Prescription Sequences")
  plt.show()

  return df

medA = see_kmeans("medA")
medB = see_kmeans("medB")
see_assumption(medA)
see_assumption(medB)