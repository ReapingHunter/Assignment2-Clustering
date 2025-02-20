import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

# np.random.seed(42)

# num_patients = 500

# # Simulated data
# df = pd.DataFrame({
#   "pnr": np.random.randint(1000, 1010, size=num_patients),  # Patient IDs
#   "eksd": pd.date_range(start="2020-01-01", periods=num_patients, freq="D"),  # Prescription dates
#   "perday": np.random.choice([1, 2, 3], size=num_patients),  # Daily dosage
#   "ATC": np.random.choice(["medA", "medB", "medC"], size=num_patients),  # Drug type
#   "dur_original": np.random.randint(7, 30, size=num_patients)  # Duration of medication
# })
script_dir = os.path.dirname(os.path.abspath(__file__)) 
file_path = os.path.join(script_dir, "med_events.csv") 
med_events = pd.read_csv(file_path)
ExamplePats = med_events.copy()
tidy = ExamplePats.copy()
tidy.columns = ["pnr", "eksd", "perday", "ATC", "dur_original"]
tidy['eksd'] = pd.to_datetime(tidy['eksd'], format='%m/%d/%Y')

def see_kmeans(arg1):

  # 1. Compute Event Intervals
  C09CA01 = tidy[tidy["ATC"] == arg1].copy()
  C09CA01 = C09CA01.sort_values(by=["pnr", "eksd"])
  C09CA01["prev_eksd"] = C09CA01.groupby("pnr")["eksd"].shift(1)
  C09CA01 = C09CA01.dropna(subset=["prev_eksd"])
  C09CA01["event_interval"] = (C09CA01["eksd"] - C09CA01["prev_eksd"]).dt.days


  # 2. Construct ECDF based on event intervals
  ecdf = C09CA01["event_interval"].sort_values().reset_index(drop=True)
  y = np.arange(1, len(ecdf) + 1) / len(ecdf)
  df_ecdf = pd.DataFrame({'x': ecdf, 'y': y})
  df_ecdf = df_ecdf[df_ecdf['y'] <= 0.8]

  plt.figure(figsize=(10, 5))
  plt.subplot(1, 2, 1)
  plt.plot(df_ecdf['x'], df_ecdf['y'])
  plt.title("80% ECDF")
  plt.subplot(1, 2, 2)
  plt.plot(ecdf, y)
  plt.title("100% ECDF")
  plt.show()
  
  # 3. Density Estimation
  density = gaussian_kde(np.log(df_ecdf['x']))
  x_vals = np.linspace(min(np.log(df_ecdf['x'])), max(np.log(df_ecdf['x'])), 100)
  y_vals = density(x_vals)
  
  plt.figure(figsize=(6, 4))
  plt.plot(x_vals, y_vals)
  plt.title("Log(event interval) Density")
  plt.show()

  # 4. Clustering
  scaler = StandardScaler()
  scaled_data = scaler.fit_transform(df_ecdf[['x']])
  silhouette_scores = []
  for k in range(2, min(10, len(scaled_data))):
    km = KMeans(n_clusters=k, random_state=1234)
    km.fit(scaled_data)
    silhouette_scores.append(km.inertia_)
    
  optimal_k = np.argmin(silhouette_scores) + 2
  km = KMeans(n_clusters=optimal_k, random_state=1234)
  df_ecdf["cluster"] = km.fit_predict(df_ecdf[['x']])

  # 5. Compute cluster stats
  cluster_summary = df_ecdf.groupby("cluster")['x'].agg(["min", "max", "median"]).reset_index()
  C09CA01 = C09CA01.merge(cluster_summary, how="left", left_on="event_interval", right_on="min")

  return C09CA01

def see_assumption(df):
  df = df.sort_values(by=["pnr", "eksd"])
  df["prev_eksd"] = df.groupby("pnr")["eksd"].shift(1)
  df = df.dropna(subset=["prev_eksd"])
  df["Duration"] = (df["eksd"] - df["prev_eksd"]).dt.days
  df["p_number"] = df.groupby("pnr").cumcount() + 1

  plt.figure(figsize=(10, 5))
  sns.boxplot(x=df["p_number"], y=df["Duration"])
  plt.axhline(df.groupby('pnr')["Duration"].median().median(), linestyle="dashed", color="red")
  plt.title("Boxplot of Event Durations")
  plt.show()

medA = see_kmeans("medA")
medB = see_kmeans("medB")
see_assumption(medA)
see_assumption(medB)