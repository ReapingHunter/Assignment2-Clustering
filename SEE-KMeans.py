import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
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
    
    # Silhouette analysis to choose optimal number of clusters
    best_k = None
    best_score = -1
    scores = {}
    for k in range(2, 10):
        km = KMeans(n_clusters=k, random_state=1234, n_init=10)
        km.fit(a_scaled)
        score = silhouette_score(a_scaled, km.labels_)
        scores[k] = score
        if score > best_score:
            best_score = score
            best_k = k
    # Plot silhouette scores
    plt.plot(list(scores.keys()), list(scores.values()), marker='o')
    plt.title("Silhouette Analysis")
    plt.xlabel("Number of clusters")
    plt.ylabel("Silhouette score")
    plt.show()
    max_cluster = best_k
    
    # K-means clustering on dfper['x']
    km_final = KMeans(n_clusters=max_cluster, random_state=1234, n_init=10)
    km_final.fit(dfper[['x']])
    dfper['cluster'] = km_final.labels_
    # Summarize log(x) by cluster: compute min and max per cluster
    ni2 = dfper.groupby('cluster')['x'].apply(lambda s: np.log(s).min())
    ni3 = dfper.groupby('cluster')['x'].apply(lambda s: np.log(s).max())
    ni2_df = ni2.reset_index().rename(columns={'x': 'Results'})
    ni3_df = ni3.reset_index().rename(columns={'x': 'Results'})
    ni2_df['Results'] = ni2_df['Results'].replace(-np.inf, 0)
    # Combine ni2 and ni3 into one DataFrame (preserving order)
    nif = pd.concat([ni2_df, ni3_df.iloc[:, 1]], axis=1)
    nif.columns = ['Cluster', 'Results', 'Results_1']
    # Remove the third column (as in R code)
    nif = nif[['Cluster', 'Results']]
    # Exponentiate the results
    nif['Results'] = np.exp(nif['Results'])
    nif['Results_1'] = np.exp(ni3_df['Results'])
    # Compute median of log(x) per cluster
    ni4 = dfper.groupby('cluster')['x'].apply(lambda s: np.median(np.log(s)))
    ni4_df = ni4.reset_index().rename(columns={'x': 'Results'})
    ni4_df = ni4_df.rename(columns={'cluster': 'Cluster'})
    nif = pd.merge(nif, ni4_df, on='Cluster')
    nif.columns = ['Cluster', 'Minimum', 'Maximum', 'Median']
    nif['Median'] = nif['Median'].replace(-np.inf, 0)
    nif = nif[nif['Median'] > 0]
    
    # Cross join Drug_see_p1 with nif
    Drug_see_p1['key'] = 1
    nif['key'] = 1
    results = pd.merge(Drug_see_p1, nif, on='key')
    results.drop('key', axis=1, inplace=True)
    # Create Final_cluster: if event.interval is between Minimum and Maximum, assign Cluster; else NaN
    # Note: In the merge, assume that the column from nif corresponding to Cluster is named 'Cluster_y'
    # Adjust column names accordingly:
    if 'Cluster_y' not in results.columns:
        # Rename the nif Cluster column after merge if needed
        results = results.rename(columns={'Cluster_x': 'Cluster', 'Cluster_y': 'Cluster'})
    results['Final_cluster'] = np.where(
        (results['event.interval'] >= results['Minimum']) & (results['event.interval'] <= results['Maximum']),
        results['Cluster'], np.nan
    )
    results = results[~results['Final_cluster'].isna()].copy()
    results['Median'] = np.exp(results['Median'])
    results = results[['pnr', 'Median', 'Cluster']].copy()
    
    # Create frequency table for results.Cluster and get the most frequent cluster
    t1 = results['Cluster'].value_counts().reset_index()
    t1.columns = ['Cluster', 'Freq']
    t1 = t1.sort_values(by='Freq', ascending=False)
    most_freq = t1.iloc[0]['Cluster']
    t1 = pd.DataFrame({'Cluster': [most_freq]})
    t1['Cluster'] = t1['Cluster'].astype(float)
    results['Cluster'] = results['Cluster'].astype(float)
    t1_merged = pd.merge(t1, results, on='Cluster', how='inner')
    t1_merged = t1_merged.iloc[[0], :]
    if 'Freq' in t1_merged.columns:
        t1_merged = t1_merged.drop(columns=['Freq'])
    t1 = t1_merged.copy()
    # Merge Drug_see_p1 with results on 'pnr'
    Drug_see_p1 = pd.merge(Drug_see_p1, results, on='pnr', how='left')
    Drug_see_p1['Median'] = Drug_see_p1['Median'].fillna(t1.iloc[0]['Median'])
    Drug_see_p1['Cluster'] = Drug_see_p1['Cluster'].fillna(0)
    Drug_see_p1['event.interval'] = Drug_see_p1['event.interval'].astype(float)
    Drug_see_p1['test'] = np.round(Drug_see_p1['event.interval'] - Drug_see_p1['Median'], 1)
    
    Drug_see_p3 = Drug_see_p1[['pnr', 'Median', 'Cluster']].copy()
    
    # Assign Duration by merging Drug_see_p0 with Drug_see_p3 on 'pnr'
    Drug_see_p0 = pd.merge(Drug_see_p0, Drug_see_p3, on='pnr', how='left')
    Drug_see_p0['Median'] = pd.to_numeric(Drug_see_p0['Median'], errors='coerce')
    Drug_see_p0['Median'] = Drug_see_p0['Median'].fillna(t1.iloc[0]['Median'])
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
    medians_of_medians = Drug_see2.groupby('pnr')['Duration'].median().median()
    
    # Plot with median reference line
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='p_number', y='Duration', data=Drug_see2)
    plt.axhline(medians_of_medians, linestyle='dashed', color='red')
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
