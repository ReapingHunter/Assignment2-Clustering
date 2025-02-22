import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
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
    max_x = dfper['x'].max()
    
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
    
    ni = max_x
    # Fixed: Use Drug_see_p1 instead of undefined variable
    Drug_see_p2 = Drug_see_p1[Drug_see_p1['event.interval'] <= ni].copy()
    
    # Compute the density of log(event.interval)
    log_event = np.log(Drug_see_p2['event.interval'].astype(float))
    density = gaussian_kde(log_event)
    x1 = np.linspace(log_event.min(), log_event.max(), 100)
    y1 = density(x1)
    plt.plot(x1, y1)
    plt.title("Log(event interval)")
    plt.show()
    z1 = x1.max()
    
    # Create a DataFrame 'a' from the density estimates and scale it
    a_df = pd.DataFrame({'x': x1, 'y': y1})
    scaler = StandardScaler()
    a_scaled = scaler.fit_transform(a_df)
    a_scaled = pd.DataFrame(a_scaled, columns=['x', 'y'])

    # Silhouette analysis to choose optimal eps
    best_eps = None
    best_score = -1
    scores = {}

    # Try different eps values
    eps_values = np.linspace(0.1, 2.0, 20)  # Adjust range as needed

    for eps in eps_values:
        db = DBSCAN(eps=eps, min_samples=5).fit(a_scaled)
        
        # Only compute silhouette score if at least 2 clusters exist
        if len(set(db.labels_)) > 1:
            score = silhouette_score(a_scaled, db.labels_)
            scores[eps] = score
            if score > best_score:
                best_score = score
                best_eps = eps

    # Plot silhouette scores
    plt.plot(list(scores.keys()), list(scores.values()), marker='o')
    plt.title("DBSCAN Silhouette Analysis")
    plt.xlabel("Epsilon (eps)")
    plt.ylabel("Silhouette Score")
    plt.show()
    optimal_eps = best_eps    
    
    # DBSCAN clustering on dfper['x']
    db = DBSCAN(eps=optimal_eps, min_samples=5)
    db.fit(dfper[['x']])
    dfper['cluster'] = db.labels_

    # Noise Removal
    dfper = dfper[dfper['cluster'] != -1]

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
    
    # Create sequence numbers per patient
    arg1['p_number'] = arg1.groupby('pnr').cumcount() + 1
    
    # Filter p_number >= 2
    Drug_see2 = arg1[arg1['p_number'] >= 2].copy()
    Drug_see2 = Drug_see2[['pnr', 'eksd', 'prev_eksd', 'p_number']]
    
    # Calculate Duration
    Drug_see2['Duration'] = (Drug_see2['eksd'] - Drug_see2['prev_eksd']).dt.days.astype(float)
    Drug_see2['p_number'] = Drug_see2['p_number'].astype(str)
    
    # Compute median duration per patient
    global_median = Drug_see2['Duration'].median()
    
    # Plot with median reference line
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='p_number', y='Duration', data=Drug_see2)
    plt.axhline(global_median, linestyle='dashed', color='red')
    plt.yticks(np.arange(0, 350, 100))
    plt.xlabel("p_number")
    plt.ylabel("Duration")
    plt.title("Boxplot of Duration by p_number")
    plt.show()
    
    return plt

# Generate medA and medB using the See() function
medA = See("medA")
medB = See("medB")

# Run the assumption plots
see_assumption(medA)
see_assumption(medB)
