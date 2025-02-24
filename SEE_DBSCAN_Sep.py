import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.distributions.empirical_distribution import ECDF
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri

# -------------------------
# Load dataset – uses AdhereR's med.events
# -------------------------
pandas2ri.activate()
robjects.r('library(AdhereR)')
med_events_r = robjects.r('med.events')
med_events = pandas2ri.rpy2py(med_events_r)
ExamplePats = med_events.copy()
tidy = ExamplePats.copy()
tidy.columns = ["pnr", "eksd", "perday", "ATC", "dur_original"]
tidy['eksd'] = pd.to_datetime(tidy['eksd'], format='%m/%d/%Y')

def patient_sampling(drug_see):
    drug_see_p1 = drug_see.sort_values(['pnr', 'eksd']).copy()
    drug_see_p1['prev_eksd'] = drug_see_p1.groupby('pnr')['eksd'].shift(1)
    drug_see_p1 = drug_see_p1.dropna(subset=['prev_eksd']).copy()
    drug_see_p1 = (drug_see_p1
                   .groupby('pnr', group_keys=False)
                   .apply(lambda x: x.sample(1, random_state=1234))
                   .reset_index(drop=True))
    drug_see_p1 = drug_see_p1[['pnr', 'eksd', 'prev_eksd']]
    drug_see_p1['event.interval'] = (drug_see_p1['eksd'] - drug_see_p1['prev_eksd']).dt.days.astype(float)
    return drug_see_p1

def ecdf_computation(drug_see):
    ecdf_see = ECDF(drug_see['event.interval'].values)
    x_vals = np.sort(drug_see['event.interval'].values)
    y_vals = ecdf_see(x_vals)
    df_ecdf = pd.DataFrame({'x': x_vals, 'y': y_vals})
    df_ecdf_80 = df_ecdf[df_ecdf['y'] <= 0.8]
    ni = df_ecdf_80['x'].max()
    return df_ecdf_80, df_ecdf, ni

def plot_ecdf(df_ecdf_80, df_ecdf):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].plot(df_ecdf_80['x'], df_ecdf_80['y'], marker='.', linestyle='none')
    axs[0].set_title("80% ECDF")
    axs[0].set_xlabel("Event Interval")
    axs[0].set_ylabel("ECDF")
    
    axs[1].plot(df_ecdf['x'], df_ecdf['y'], marker='.', linestyle='none')
    axs[1].set_title("100% ECDF")
    axs[1].set_xlabel("Event Interval")
    axs[1].set_ylabel("ECDF")
    plt.show()

def plot_frequency_table(drug_see):
    event_counts = drug_see['pnr'].value_counts().sort_index()
    xticks = [1, 15, 32, 48, 64, 90, 96]
    plt.figure(figsize=(8, 5))
    plt.bar(event_counts.index, event_counts.values)
    plt.xticks(xticks)
    plt.xlabel("pnr")
    plt.ylabel("Frequency")
    plt.title("Frequency of pnr (Adjusted)")
    plt.show()

def density_estimation(drug_see):
    log_intervals = np.log(drug_see['event.interval'])
    kde = gaussian_kde(log_intervals)
    x_grid = np.linspace(log_intervals.min(), log_intervals.max(), 100)
    y_kde = kde(x_grid)
    return x_grid, y_kde

def plot_density(x_grid, y_kde):
    plt.figure(figsize=(8, 5))
    plt.plot(x_grid, y_kde)
    plt.title("Density of log(event interval)")
    plt.xlabel("log(event interval)")
    plt.ylabel("Density")
    plt.show()

def silhouette_analysis(scaled_data, min_samples_dbscan, eps_range):
    silhouette_scores = {}
    for eps in eps_range:
        dbscan_model = DBSCAN(eps=eps, min_samples=min_samples_dbscan)
        labels = dbscan_model.fit_predict(scaled_data)
        # Exclude noise points (label == -1)
        mask = labels != -1
        unique_labels = np.unique(labels[mask])
        if len(unique_labels) < 2:
            continue  # Skip if less than 2 clusters (silhouette score undefined)
        score = silhouette_score(scaled_data[mask], labels[mask])
        silhouette_scores[eps] = score
    if silhouette_scores:
        best_eps = max(silhouette_scores, key=silhouette_scores.get)
        plot_silhouette(silhouette_scores)
    else:
        best_eps = eps_range[0]
    return best_eps
    
def plot_silhouette(silhouette_scores):
    plt.figure(figsize=(8, 5))
    plt.plot(list(silhouette_scores.keys()), list(silhouette_scores.values()), marker='o')
    plt.title("Silhouette Analysis")
    plt.xlabel("eps value")
    plt.ylabel("Silhouette Score")
    plt.show()

def dbscan_cluster(df_ecdf, scaled_data, best_eps, min_samples_dbscan):
    db = DBSCAN(eps=best_eps, min_samples=min_samples_dbscan)
    labels = db.fit_predict(scaled_data)
    df_ecdf['cluster'] = labels  # DBSCAN labels noise as -1

def cluster_assignment(df_ecdf, drug_see_p0, drug_see_p1):
    # For each cluster, compute the min, max, and median on the log scale then exponentiate
    cluster_stats = (df_ecdf[df_ecdf['cluster'] != -1].groupby('cluster')['x']
                     .agg(min_log=lambda x: np.log(x).min(),
                          max_log=lambda x: np.log(x).max(),
                          median_log=lambda x: np.log(x).median())
                     .reset_index())
    cluster_stats['Minimum'] = np.exp(cluster_stats['min_log'])
    cluster_stats['Maximum'] = np.exp(cluster_stats['max_log'])
    cluster_stats['Median'] = np.exp(cluster_stats['median_log'])
    cluster_stats = cluster_stats[cluster_stats['Median'] > 0]
    
    # Cross join drug_see_p1 with cluster_stats
    drug_see_p1['_key'] = 1
    cluster_stats['_key'] = 1
    cross_df = pd.merge(drug_see_p1, cluster_stats, on='_key').drop('_key', axis=1)
    
    # For each row, assign Final_cluster if the event interval falls within the cluster’s [Minimum, Maximum]
    cross_df['Final_cluster'] = cross_df.apply(
        lambda row: row['cluster'] if (row['event.interval'] >= row['Minimum'] and row['event.interval'] <= row['Maximum'])
        else np.nan, axis=1)
    results = cross_df.dropna(subset=['Final_cluster']).copy()
    results = results[['pnr', 'Median', 'Final_cluster']]
    
    # Determine the most common cluster
    most_common_cluster = results['Final_cluster'].value_counts().idxmax()
    default_median = cluster_stats.loc[cluster_stats['cluster'] == most_common_cluster, 'Median'].values[0]
    
    # Merge cluster assignments back into drug_see_p1
    drug_see_p1 = pd.merge(drug_see_p1, results, on='pnr', how='left')
    drug_see_p1['Median'] = drug_see_p1['Median'].fillna(default_median)
    drug_see_p1['Cluster'] = drug_see_p1['Final_cluster'].fillna(0)
    drug_see_p1['test'] = (drug_see_p1['event.interval'] - drug_see_p1['Median']).round(1)
    
    drug_see_p3 = drug_see_p1[['pnr', 'Median', 'Cluster']]
    
    # Merge these assignments back into the original filtered data (drug_see_p0)
    final_df = pd.merge(drug_see_p0, drug_see_p3, on='pnr', how='left')
    final_df['Median'] = final_df['Median'].fillna(default_median)
    final_df['Cluster'] = final_df['Cluster'].fillna(0)
    
    return final_df

def see(arg1, min_samples_dbscan=2, eps_range=np.linspace(0.001, 10.0, 100)):
    # Filter rows where ATC equals arg1
    C09CA01 = tidy[tidy['ATC'] == arg1].copy()
    drug_see_p0 = C09CA01.copy()
    
    drug_see_p1 = patient_sampling(C09CA01)
    
    df_ecdf_80, df_ecdf, ni = ecdf_computation(drug_see_p1)
    
    plot_ecdf(df_ecdf_80, df_ecdf)
    plot_frequency_table(drug_see_p1)
    
    drug_see_p2 = drug_see_p1[drug_see_p1['event.interval'] <= ni].copy()
    
    x_grid, y_kde = density_estimation(drug_see_p2)
    plot_density(x_grid, y_kde)
    
    # Instead of clustering on the density grid, cluster on the ECDF x values.
    scaler_ecdf = StandardScaler()
    ecdf_scaled = scaler_ecdf.fit_transform(df_ecdf[['x']])
    
    best_eps = silhouette_analysis(ecdf_scaled, min_samples_dbscan, eps_range)
    dbscan_cluster(df_ecdf, ecdf_scaled, best_eps, min_samples_dbscan)
    
    final_df = cluster_assignment(df_ecdf, drug_see_p0, drug_see_p1)
    return final_df

def see_assumption(df):
    df_sorted = df.sort_values(['pnr', 'eksd']).copy()
    df_sorted['prev_eksd'] = df_sorted.groupby('pnr')['eksd'].shift(1)
    df_sorted['p_number'] = df_sorted.groupby('pnr').cumcount() + 1
    df2 = df_sorted[df_sorted['p_number'] >= 2].copy()
    df2 = df2[['pnr', 'eksd', 'prev_eksd', 'p_number']]
    df2['Duration'] = (df2['eksd'] - df2['prev_eksd']).dt.days
    df2['p_number'] = df2['p_number'].astype(str)
    
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='p_number', y='Duration', data=df2)
    medians_of_medians = df2['Duration'].median()
    plt.axhline(y=medians_of_medians, color='red', linestyle='--')
    plt.title("Duration by Prescription Number with Median Line")
    plt.xlabel("Prescription Number")
    plt.ylabel("Duration (days)")
    plt.show()

# medA = see("medA")
# medB = see("medB")

# see_assumption(medA)
# see_assumption(medB)
