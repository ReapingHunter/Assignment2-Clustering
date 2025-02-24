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

arg1 = "medA"

def see(med_type, min_samples_dbscan=2, eps_range=np.linspace(0.001, 10.0, 100)):
    # Filter data for the medication type (e.g., "medA")
    data_med = tidy[tidy['ATC'] == med_type].copy()
    drug_see_p0 = data_med.copy()  # Original filtered data

    # Sort by patient and date; compute previous prescription date per patient.
    drug_see_p1 = data_med.sort_values(['pnr', 'eksd']).copy()
    drug_see_p1['prev_eksd'] = drug_see_p1.groupby('pnr')['eksd'].shift(1)
    drug_see_p1 = drug_see_p1.dropna(subset=['prev_eksd']).copy()

    # For each patient, randomly sample one consecutive prescription pair.
    drug_see_p1 = (drug_see_p1.groupby('pnr', group_keys=False)
                   .apply(lambda x: x.sample(1, random_state=1234))
                   .reset_index(drop=True))
    drug_see_p1 = drug_see_p1[['pnr', 'eksd', 'prev_eksd']]

    # Compute event interval (in days)
    drug_see_p1['event.interval'] = (drug_see_p1['eksd'] - drug_see_p1['prev_eksd']).dt.days.astype(float)

    # Compute the ECDF function using statsmodels
    ecdf_func = ECDF(drug_see_p1['event.interval'].values)

    # Use sorted event intervals as x-values
    x_vals = np.sort(drug_see_p1['event.interval'].values)
    y_vals = ecdf_func(x_vals)

    dfper = pd.DataFrame({'x': x_vals, 'y': y_vals})

    # Retain the lower 80% of the ECDF (i.e. cumulative probability <= 0.8)
    dfper_80 = dfper[dfper['y'] <= 0.8]
    ni = dfper_80['x'].max()

    # Plot the 80% and full ECDF
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].plot(dfper_80['x'], dfper_80['y'], marker='.', linestyle='none')
    axs[0].set_title("80% ECDF")
    axs[0].set_xlabel("Event Interval")
    axs[0].set_ylabel("ECDF")

    axs[1].plot(dfper['x'], dfper['y'], marker='.', linestyle='none')
    axs[1].set_title("100% ECDF")
    axs[1].set_xlabel("Event Interval")
    axs[1].set_ylabel("ECDF")
    plt.show()

    # Plot frequency of events per patient
    plt.figure(figsize=(8, 5))
    drug_see_p1['pnr'].value_counts().plot(kind='bar')
    plt.title("Frequency of Events per Patient")
    plt.xlabel("pnr")
    plt.ylabel("Count")
    plt.show()

    # Subset data to include only event intervals up to the 80th percentile.
    drug_see_p2 = drug_see_p1[drug_see_p1['event.interval'] <= ni].copy()

    # --- Density Estimation on log(event.interval) ---
    log_intervals = np.log(drug_see_p2['event.interval'])
    kde = gaussian_kde(log_intervals)
    x_grid = np.linspace(log_intervals.min(), log_intervals.max(), 100)
    y_kde = kde(x_grid)

    plt.figure(figsize=(8, 5))
    plt.plot(x_grid, y_kde)
    plt.title("Density of log(event interval)")
    plt.xlabel("log(event interval)")
    plt.ylabel("Density")
    plt.show()

    # =============================================================================
    # 3. Silhouette Analysis to Choose eps for DBSCAN
    # =============================================================================
    # We'll cluster using the event interval values from the ECDF.
    X = dfper[['x']].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    silhouette_scores = {}
    for eps in eps_range:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples_dbscan)
        labels = dbscan.fit_predict(X_scaled)
        # Exclude noise points (label == -1)
        mask = labels != -1
        unique_labels = np.unique(labels[mask])
        if len(unique_labels) < 2:
            continue  # Skip if less than 2 clusters (silhouette score undefined)
        score = silhouette_score(X_scaled[mask], labels[mask])
        silhouette_scores[eps] = score

    if silhouette_scores:
        best_eps = max(silhouette_scores, key=silhouette_scores.get)
        # Plot silhouette score vs eps
        eps_vals = list(silhouette_scores.keys())
        scores = list(silhouette_scores.values())
        plt.figure(figsize=(8, 5))
        plt.plot(eps_vals, scores, marker='o')
        plt.title("Silhouette Score vs eps for DBSCAN")
        plt.xlabel("eps")
        plt.ylabel("Silhouette Score")
        plt.show()
    else:
        best_eps = eps_range[0]

    # =============================================================================
    # 4. DBSCAN Clustering with Chosen eps
    # =============================================================================
    db = DBSCAN(eps=best_eps, min_samples=min_samples_dbscan)
    labels = db.fit_predict(X_scaled)
    dfper['cluster'] = labels  # DBSCAN labels noise as -1

    cluster_stats = (dfper[dfper['cluster'] != -1].groupby('cluster')['x'] # Noise filter
                     .agg(min_log=lambda x: np.log(x).min(),
                          max_log=lambda x: np.log(x).max(),
                          median_log=lambda x: np.log(x).median())
                     .reset_index())
    cluster_stats['Minimum'] = np.exp(cluster_stats['min_log'])
    cluster_stats['Maximum'] = np.exp(cluster_stats['max_log'])
    cluster_stats['Median'] = np.exp(cluster_stats['median_log'])
    # (Keep only clusters with a positive median; typically all if intervals > 0)
    cluster_stats = cluster_stats[cluster_stats['Median'] > 0]

    # =============================================================================
    # 5. Assign Clusters to Patients via Cross Join
    # =============================================================================
    drug_see_p1['key'] = 1
    cluster_stats['key'] = 1
    cross_df = pd.merge(drug_see_p1, cluster_stats, on='key').drop('key', axis=1)

    # For each row, assign Final_cluster if event.interval falls within the cluster’s [Minimum, Maximum]
    cross_df['Final_cluster'] = cross_df.apply(
        lambda row: row['cluster'] if (row['event.interval'] >= row['Minimum'] and row['event.interval'] <= row['Maximum'])
        else np.nan, axis=1)
    results = cross_df.dropna(subset=['Final_cluster']).copy()
    results = results[['pnr', 'Median', 'Final_cluster']]

    # Determine the most common cluster
    most_common_cluster = results['Final_cluster'].value_counts().idxmax()
    default_median = cluster_stats.loc[cluster_stats['cluster'] == most_common_cluster, 'Median'].values[0]

    # Merge cluster assignment back into drug_see_p1.
    drug_see_p1 = pd.merge(drug_see_p1, results, on='pnr', how='left')
    drug_see_p1['Median'] = drug_see_p1['Median'].fillna(default_median)
    drug_see_p1['Cluster'] = drug_see_p1['Final_cluster'].fillna(0)
    drug_see_p1['test'] = (drug_see_p1['event.interval'] - drug_see_p1['Median']).round(1)
    
    drug_see_p3 = drug_see_p1[['pnr', 'Median', 'Cluster']]

    # Merge the cluster assignments back into the original filtered data.
    final_df = pd.merge(drug_see_p0, drug_see_p3, on='pnr', how='left')
    final_df['Median'] = final_df['Median'].fillna(default_median)
    final_df['Cluster'] = final_df['Cluster'].fillna(0)

    return final_df

# =============================================================================
# 6. Assumption Check Function
# =============================================================================
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

# =============================================================================
# 7. Example Usage
# =============================================================================
medA = see("medA")
medB = see("medB")

see_assumption(medA)
see_assumption(medB)
