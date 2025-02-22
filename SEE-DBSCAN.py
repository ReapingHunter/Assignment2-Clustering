import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import os
# -------------------------
# Load dataset – equivalent to R's med.events
# -------------------------

script_dir = os.path.dirname(os.path.abspath(__file__)) 
file_path = os.path.join(script_dir, "med_events.csv")  
med_events = pd.read_csv(file_path)  # Ensure the CSV file is in the working directory
ExamplePats = med_events.copy()
tidy = ExamplePats.copy()
tidy.columns = ["pnr", "eksd", "perday", "ATC", "dur_original"]
tidy['eksd'] = pd.to_datetime(tidy['eksd'], format='%m/%d/%Y')

arg1 = "medA"

def optimal_eps(data):
    """Find the optimal epsilon using a k-distance plot."""
    neighbors = NearestNeighbors(n_neighbors=5)
    neighbors_fit = neighbors.fit(data)
    distances, _ = neighbors_fit.kneighbors(data)
    distances = np.sort(distances[:, -1])
    return distances[int(0.9 * len(distances))]

def See(arg1):
    # Filter rows where ATC equals arg1
    C09CA01 = tidy[tidy['ATC'] == arg1].copy()
    # Take a random sequence of consecutive prescription in the dataset
    Drug_see_p0 = C09CA01.copy()
    Drug_see_p1 = C09CA01.copy()
    # Sort by pnr and eksd and compute previous prescription date per patient
    Drug_see_p1 = Drug_see_p1.sort_values(by=['pnr', 'eksd'])
    Drug_see_p1['prev_eksd'] = Drug_see_p1.groupby('pnr')['eksd'].shift(1)
    Drug_see_p1 = Drug_see_p1.dropna(subset=['prev_eksd'])
    # For each patient, randomly sample one row (fixing deprecation warning by leaving grouping columns)
    Drug_see_p1 = Drug_see_p1.groupby('pnr', group_keys=False).apply(lambda x: x.sample(n=1, random_state=1234))
    Drug_see_p1 = Drug_see_p1[['pnr', 'eksd', 'prev_eksd']].copy()
    # Compute event.interval as the duration (in days) between prescriptions
    Drug_see_p1['event.interval'] = (Drug_see_p1['eksd'] - Drug_see_p1['prev_eksd']).dt.days.astype(float)
    
    # Generate the ECDF of event.interval
    ecdf_func = ECDF(Drug_see_p1['event.interval'])
    x_vals = ecdf_func.x
    y_vals = ecdf_func.y
    dfper = pd.DataFrame({'x': x_vals, 'y': y_vals})
    
    # Retain the 20% of the ECDF (remove the upper 20%)
    dfper = dfper[dfper['y'] <= 0.8]
    # Remove any non-finite x values (fixes the infinity error)
    dfper = dfper[np.isfinite(dfper['x'])]
    
    # Plot the 80% and 100% ECDF side by side
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].plot(dfper['x'], dfper['y'])
    axs[0].set_title("80% ECDF")
    axs[1].plot(x_vals, y_vals)
    axs[1].set_title("100% ECDF")
    plt.show()
    
    # Create and plot a frequency table for pnr
    m1 = Drug_see_p1['pnr'].value_counts()
    m1.plot(kind='bar', title="Frequency of pnr")
    plt.show()
    
    # DBSCAN clustering on dfper['x']
    dfper_scaled = StandardScaler().fit_transform(dfper[['x']])
    eps_value = optimal_eps(dfper_scaled)
    db = DBSCAN(eps=eps_value, min_samples=5).fit(dfper_scaled)
    dfper['cluster'] = db.labels_

    # Process clusters
    cluster_stats = dfper.groupby('cluster')['x'].agg(['min', 'max', 'median']).reset_index()
    cluster_stats.columns = ['Cluster', 'Minimum', 'Maximum', 'Median']
    
    # Assign clusters
    results = Drug_see_p1.copy()
    results = results.merge(cluster_stats, how='cross')
    results['Final_cluster'] = np.where(
        (results['event.interval'] >= results['Minimum']) & (results['event.interval'] <= results['Maximum']),
        results['Cluster'], np.nan)
    results = results.dropna().copy()
    results = results[['pnr', 'Median', 'Cluster']]
    
    # Merge results back to original dataset
    Drug_see_p0 = pd.merge(C09CA01, results, on='pnr', how='left')
    Drug_see_p0['Median'] = Drug_see_p0['Median'].fillna(cluster_stats['Median'].min())
    Drug_see_p0['Cluster'] = Drug_see_p0['Cluster'].fillna(0)

    return Drug_see_p0

def see_assumption(arg1):
    # Sort by pnr and eksd and compute previous date per group
    arg1 = arg1.sort_values(by=['pnr', 'eksd'])
    arg1['prev_eksd'] = arg1.groupby('pnr')['eksd'].shift(1)
    Drug_see2 = arg1.copy()
    Drug_see2['p_number'] = Drug_see2.groupby('pnr').cumcount() + 1
    Drug_see2 = Drug_see2[Drug_see2['p_number'] >= 2].copy()
    Drug_see2 = Drug_see2[['pnr', 'eksd', 'prev_eksd', 'p_number']].copy()
    # Convert Duration to numeric (in days)
    Drug_see2['Duration'] = (Drug_see2['eksd'] - Drug_see2['prev_eksd']).dt.days.astype(float)
    Drug_see2['p_number'] = Drug_see2['p_number'].astype(str)
    
    # Create boxplot using seaborn
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='p_number', y='Duration', data=Drug_see2)
    plt.title("Boxplot of Duration by p_number")
    plt.show()
    
    medians_of_medians = Drug_see2.groupby('pnr')['Duration'].median().reset_index().rename(columns={'Duration': 'median_duration'})
    
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='p_number', y='Duration', data=Drug_see2)
    global_median = medians_of_medians['median_duration'].mean()
    plt.axhline(global_median, linestyle='dashed', color='red')
    plt.title("Boxplot of Duration with Median Line")
    plt.show()
    
    return plt

# Generate medA and medB using the See() function
medA = See("medA")
medB = See("medB")

# Run the assumption plots
see_assumption(medA)
see_assumption(medB)
