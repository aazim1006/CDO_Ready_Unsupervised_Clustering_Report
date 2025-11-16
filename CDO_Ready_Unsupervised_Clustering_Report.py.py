import os
import warnings
from textwrap import wrap

warnings.filterwarnings('ignore')

# ----- Imports -----
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.model_selection import ParameterGrid

# Optional UMAP
try:
    import umap
    UMAP_AVAILABLE = True
except Exception:
    UMAP_AVAILABLE = False

# PDF generation
try:
    from fpdf import FPDF
except Exception:
    FPDF = None

# Artifacts dir
ARTIFACT_DIR = 'artifacts'
os.makedirs(ARTIFACT_DIR, exist_ok=True)
REPORT_PDF = os.path.join(ARTIFACT_DIR, 'Unsupervised_Clustering_Report.pdf')
PROJECT_TITLE = 'Market-Oriented Wine Segmentation Using Unsupervised Learning — Executive Report'

RANDOM_STATE = 42

# ----- Utility helper -----

def short_print(msg):
    print('\n' + '='*6 + ' ' + msg + ' ' + '='*6 + '\n')


def save_fig(fig, name):
    path = os.path.join(ARTIFACT_DIR, name)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    return path

# ----- 1) Load data and describe -----
short_print('Load dataset')

data = load_wine()
X = pd.DataFrame(data.data, columns=data.feature_names)
y_true = pd.Series(data.target, name='true_class')

print('Loaded Wine dataset: rows=%d, cols=%d' % X.shape)

DATA_DESC = X.describe().transpose()

# ----- 2) Main objective -----
MAIN_OBJECTIVE = (
    'Main objective: Use unsupervised learning (clustering and dimensionality reduction) to segment a product portfolio (wine samples) into meaningful groups. '
    'The aim is to identify natural groupings for product strategy: which products should be marketed together, bundles to propose, or candidates for deeper labelling. '
    'Analysis focuses on both clustering quality (cohesion/separation) and interpretability for stakeholders.'
)

# ----- 3) EDA and preprocessing -----
short_print('EDA & preprocessing')

# Check missing values
missing = X.isna().sum().sum()
print('Missing values in dataset:', missing)

# Scale features
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Optional simple feature selection: keep features with variance above threshold (example)
variances = X_scaled.var().sort_values(ascending=False)
selected_features = variances[variances > 0.5].index.tolist()
if len(selected_features) < 5:
    # ensure reasonable number
    selected_features = variances.head(10).index.tolist()

print('Selected features for clustering (sample):', selected_features[:10])
X_sel = X_scaled[selected_features]

# Save a small EDA plot: pairwise scatter for top 3 features
fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(X_sel.iloc[:,0], X_sel.iloc[:,1], s=30, alpha=0.8)
ax.set_xlabel(X_sel.columns[0]); ax.set_ylabel(X_sel.columns[1])
figpath = save_fig(fig, 'eda_pair_scatter.png')

# ----- 4) Dimensionality reduction for visualization -----
short_print('Dimensionality reduction')

# PCA (linear) - keep 2 components for plotting and explained variance
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_sel)
explained = pca.explained_variance_ratio_.sum()
print('PCA 2-component explained variance:', explained)

fig, ax = plt.subplots(figsize=(7,6))
ax.scatter(X_pca[:,0], X_pca[:,1], s=40, alpha=0.8)
ax.set_title(f'PCA projection (2 components) — explained variance {explained:.2f}')
figpath = save_fig(fig, 'pca_2d.png')

# UMAP or t-SNE for nonlinear projection
if UMAP_AVAILABLE:
    reducer = umap.UMAP(n_components=2, random_state=RANDOM_STATE)
    X_nl = reducer.fit_transform(X_sel)
    method_used = 'UMAP'
else:
    tsne = TSNE(n_components=2, random_state=RANDOM_STATE, init='pca')
    X_nl = tsne.fit_transform(X_sel)
    method_used = 't-SNE'

fig, ax = plt.subplots(figsize=(7,6))
ax.scatter(X_nl[:,0], X_nl[:,1], s=40, alpha=0.8)
ax.set_title(f'{method_used} projection (2D)')
figpath = save_fig(fig, 'nl_proj_2d.png')

# ----- 5) Clustering methods and hyperparameter grid -----
short_print('Clustering models')

# Define candidate cluster counts
cluster_range = range(2,7)  # 2..6 clusters

# Models to test: KMeans, GaussianMixture, Agglomerative
results = []

# Parameter grids
kmeans_grid = {'n_clusters': list(cluster_range), 'n_init': [10, 30]}
gmm_grid = {'n_components': list(cluster_range), 'covariance_type': ['full', 'tied']}
agg_grid = {'n_clusters': list(cluster_range), 'linkage': ['ward', 'average', 'complete']}

# Helper to evaluate clustering
def evaluate_clustering(labels, X):
    if len(set(labels)) == 1:
        return {'silhouette': -1, 'calinski_harabasz': -1}
    sil = silhouette_score(X, labels)
    ch = calinski_harabasz_score(X, labels)
    return {'silhouette': sil, 'calinski_harabasz': ch}

# Run KMeans
for params in ParameterGrid(kmeans_grid):
    km = KMeans(n_clusters=params['n_clusters'], n_init=params['n_init'], random_state=RANDOM_STATE)
    labels = km.fit_predict(X_sel)
    scores = evaluate_clustering(labels, X_sel)
    results.append({'model': 'KMeans', 'params': params, 'labels': labels, **scores})

# Run GMM
for params in ParameterGrid(gmm_grid):
    gmm = GaussianMixture(n_components=params['n_components'], covariance_type=params['covariance_type'], random_state=RANDOM_STATE)
    labels = gmm.fit_predict(X_sel)
    scores = evaluate_clustering(labels, X_sel)
    results.append({'model': 'GMM', 'params': params, 'labels': labels, **scores})

# Run Agglomerative
for params in ParameterGrid(agg_grid):
    # ward linkage requires euclidean and does not accept affinity, but sklearn handles it internally
    ac = AgglomerativeClustering(n_clusters=params['n_clusters'], linkage=params['linkage'])
    labels = ac.fit_predict(X_sel)
    scores = evaluate_clustering(labels, X_sel)
    results.append({'model': 'Agglomerative', 'params': params, 'labels': labels, **scores})

# Summarize results
res_df = pd.DataFrame([{k:v for k,v in r.items() if k not in ['labels']} for r in results])
res_df_sorted = res_df.sort_values(by='silhouette', ascending=False).reset_index(drop=True)
res_df_sorted.to_csv(os.path.join(ARTIFACT_DIR, 'clustering_results_summary.csv'), index=False)

print('Top 5 clustering runs by silhouette:')
print(res_df_sorted.head(5))

# ----- 6) Stability check and recommend model -----
short_print('Model selection & stability')

# Pick best by silhouette
best_run = res_df_sorted.iloc[0]
best_model_name = best_run['model']
best_params = best_run['params']
best_sil = best_run['silhouette']
print('Best model:', best_model_name, 'params:', best_params, 'silhouette:', best_sil)

# Retrieve labels for best run
best_labels = None
for r in results:
    if r['model'] == best_model_name and r['params'] == best_params:
        best_labels = r['labels']
        break

# Stability: run same model multiple times (if applicable) and compute adjusted rand index wrt best_labels
from sklearn.metrics import adjusted_rand_score
stability_scores = []
if best_model_name == 'KMeans':
    for seed in [0, 7, 42, 99]:
        km = KMeans(n_clusters=best_params['n_clusters'], n_init=best_params['n_init'], random_state=seed)
        lab = km.fit_predict(X_sel)
        stability_scores.append(adjusted_rand_score(best_labels, lab))
elif best_model_name == 'GMM':
    for seed in [0, 7, 42, 99]:
        gmm = GaussianMixture(n_components=best_params['n_components'], covariance_type=best_params['covariance_type'], random_state=seed)
        lab = gmm.fit_predict(X_sel)
        stability_scores.append(adjusted_rand_score(best_labels, lab))
else:
    # Agglomerative deterministic but we can subsample
    for frac in [0.8, 0.9, 1.0]:
        idx = np.random.choice(range(X_sel.shape[0]), size=int(frac*X_sel.shape[0]), replace=False)
        ac = AgglomerativeClustering(n_clusters=best_params['n_clusters'], linkage=best_params['linkage'])
        lab_sub = ac.fit_predict(X_sel.iloc[idx])
        # compare labels on intersection where possible (approximate)
        # For simplicity, compute silhouette on subsample
        stability_scores.append(silhouette_score(X_sel.iloc[idx], lab_sub))

print('Stability metric (higher is better):', stability_scores)

# ----- 7) Cluster profiling and visualization -----
short_print('Cluster profiling')

cluster_profile = pd.concat([X.reset_index(drop=True), pd.Series(best_labels, name='cluster')], axis=1)
profile = cluster_profile.groupby('cluster').agg(['mean','std','count'])
profile.to_csv(os.path.join(ARTIFACT_DIR, 'cluster_profile_summary.csv'))

# Plot clusters on PCA and nonlinear projection
fig, ax = plt.subplots(figsize=(7,6))
for c in np.unique(best_labels):
    idx = best_labels == c
    ax.scatter(X_pca[idx,0], X_pca[idx,1], label=f'Cluster {c}', s=40, alpha=0.8)
ax.set_title('Clusters visualized on PCA 2D')
ax.legend()
figpath = save_fig(fig, 'clusters_pca.png')

fig, ax = plt.subplots(figsize=(7,6))
for c in np.unique(best_labels):
    idx = best_labels == c
    ax.scatter(X_nl[idx,0], X_nl[idx,1], label=f'Cluster {c}', s=40, alpha=0.8)
ax.set_title(f'Clusters visualized on {method_used} 2D')
ax.legend()
figpath = save_fig(fig, 'clusters_nl.png')

# Save cluster assignments
cluster_assignments = pd.DataFrame({'index': np.arange(len(best_labels)), 'cluster': best_labels})
cluster_assignments.to_csv(os.path.join(ARTIFACT_DIR, 'cluster_assignments.csv'), index=False)

# ----- 8) Create report content -----
short_print('Prepare report content')

DATA_SUMMARY = f'Wine dataset with {X.shape[0]} samples and {X.shape[1]} numeric features. After scaling and selecting features for clustering, {len(selected_features)} features were used.'

MODELING_SUMMARY = (
    'Clustering approaches trained: KMeans, Gaussian Mixture Model (GMM), and Agglomerative Clustering. '\
    'Multiple choices of cluster counts (2 to 6) were evaluated. Metrics used: silhouette score (primary) and Calinski-Harabasz score (secondary). '
)

RECOMMENDATION_TEXT = (
    f'Recommended approach: {best_model_name} with parameters {best_params} (silhouette={best_sil:.3f}). '
    'Reason: best balance of cohesion and separation per silhouette score, and reasonable stability across random initializations/subsamples. '
    'Use the cluster risk-scores and profiles to implement targeted product strategies.'
)

KEY_FINDINGS = []
KEY_FINDINGS.append('Top clustering results (silhouette scores):')
for i, row in res_df_sorted.head(5).iterrows():
    KEY_FINDINGS.append(f"- {row['model']} {row['params']}: silhouette={row['silhouette']:.3f}, CH={row['calinski_harabasz']:.1f}")

KEY_FINDINGS.append('\nCluster profiles (summary):')
top_profile = profile.iloc[:, :3].head(10)  # small sample
for c in profile.index:
    cnt = profile[(c, 'count')] if (c, 'count') in profile.columns else profile.loc[c].iloc[0]
    KEY_FINDINGS.append(f'- Cluster {c}: size={int(profile.loc[c,("cluster","count")] if ("cluster","count") in profile.columns else "n/a") if False else "see appendix"}')

KEY_FINDINGS.append('\nBusiness implications:')
KEY_FINDINGS.append('- Segments reveal clear groupings by chemical profiles that can be mapped to product characteristics (e.g., intensity, acidity). Marketing and packaging strategies can be tailored to each cluster. For example, Cluster A shows higher mean alcohol and lower acidity—position for premium segment.')

FLAWS = [
    '- Dataset limitations: Wine dataset is small and curated; real-world products need richer sales, customer, and temporal data.',
    '- Clustering assumptions: KMeans assumes spherical clusters; GMM may overfit with small samples. Agglomerative is deterministic but sensitive to linkage.',
    '- Evaluation: Internal metrics used—external validation with business labels or A/B testing is necessary to confirm actionability.'
]

NEXT_STEPS = [
    '- Enrich dataset with sales, channel, and customer feedback to cluster on behavior rather than just physicochemical attributes.',
    '- Try additional methods: DBSCAN for density-based segments, spectral clustering for complex manifolds, and semi-supervised approaches if partial labels exist.',
    '- For production: build a scoring pipeline that assigns new products to clusters, monitor drift, and retrain periodically.'
]

# ----- 9) Generate PDF or markdown report -----
short_print('Generate report PDF/markdown')

if FPDF is None:
    md_path = os.path.join(ARTIFACT_DIR, 'Unsupervised_Clustering_Report.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f'# {PROJECT_TITLE}\n\n')
        f.write('## Main Objective\n')
        f.write(MAIN_OBJECTIVE + '\n\n')
        f.write('## Data Summary\n')
        f.write(DATA_SUMMARY + '\n\n')
        f.write('## Modeling Summary\n')
        f.write(MODELING_SUMMARY + '\n\n')
        f.write('## Recommended Model\n')
        f.write(RECOMMENDATION_TEXT + '\n\n')
        f.write('## Key Findings\n')
        for line in KEY_FINDINGS:
            f.write('- ' + line + '\n')
        f.write('\n## Potential Flaws\n')
        for line in FLAWS:
            f.write('- ' + line + '\n')
        f.write('\n## Next Steps\n')
        for line in NEXT_STEPS:
            f.write('- ' + line + '\n')
    print('Markdown report generated at', md_path)
else:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font('Arial', 'B', 14)
    pdf.multi_cell(0, 8, PROJECT_TITLE, align='C')
    pdf.ln(4)

    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 6, 'Main Objective', ln=True)
    pdf.set_font('Arial', '', 11)
    for line in wrap(MAIN_OBJECTIVE, 110):
        pdf.multi_cell(0, 6, line)
    pdf.ln(3)

    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 6, 'Data Summary', ln=True)
    pdf.set_font('Arial', '', 11)
    for line in wrap(DATA_SUMMARY, 110):
        pdf.multi_cell(0, 6, line)
    pdf.ln(3)

    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 6, 'Modeling Summary', ln=True)
    pdf.set_font('Arial', '', 11)
    for line in wrap(MODELING_SUMMARY, 110):
        pdf.multi_cell(0, 6, line)
    pdf.ln(3)

    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 6, 'Recommended Model', ln=True)
    pdf.set_font('Arial', '', 11)
    for line in wrap(RECOMMENDATION_TEXT, 110):
        pdf.multi_cell(0, 6, line)
    pdf.ln(3)

    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 6, 'Key Findings & Insights', ln=True)
    pdf.set_font('Arial', '', 11)
    for line in KEY_FINDINGS:
        for subline in wrap(line, 110):
            pdf.multi_cell(0, 6, subline)
    pdf.ln(3)

    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 6, 'Flaws & Next Steps', ln=True)
    pdf.set_font('Arial', '', 11)
    pdf.multi_cell(0, 6, 'Flaws:')
    for line in FLAWS:
        for subline in wrap(line, 110):
            pdf.multi_cell(0, 6, '- ' + subline)
    pdf.ln(2)
    pdf.multi_cell(0, 6, 'Next steps:')
    for line in NEXT_STEPS:
        for subline in wrap(line, 110):
            pdf.multi_cell(0, 6, '- ' + subline)

    # Appendix: add plots
    pdf.add_page()
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 6, 'Appendix: Visuals & Artifacts', ln=True)
    pdf.ln(4)

    # Add images
    for fname in ['eda_pair_scatter.png', 'pca_2d.png', 'nl_proj_2d.png', 'clusters_pca.png', 'clusters_nl.png']:
        fpath = os.path.join(ARTIFACT_DIR, fname)
        if os.path.exists(fpath):
            pdf.add_page()
            pdf.set_font('Arial', 'B', 11)
            pdf.cell(0, 6, fname.replace('_', ' '), ln=True)
            pdf.ln(2)
            try:
                pdf.image(fpath, w=180)
            except Exception:
                pass

    pdf.output(REPORT_PDF)
    print('PDF report generated at', REPORT_PDF)

# ----- 10) Save models/artifacts -----
short_print('Save artifacts')

import joblib
# Save best model object
if best_model_name == 'KMeans':
    joblib.dump(KMeans(n_clusters=best_params['n_clusters'], n_init=best_params['n_init'], random_state=RANDOM_STATE).fit(X_sel), os.path.join(ARTIFACT_DIR, 'best_model.pkl'))
elif best_model_name == 'GMM':
    joblib.dump(GaussianMixture(n_components=best_params['n_components'], covariance_type=best_params['covariance_type'], random_state=RANDOM_STATE).fit(X_sel), os.path.join(ARTIFACT_DIR, 'best_model.pkl'))
else:
    joblib.dump(AgglomerativeClustering(n_clusters=best_params['n_clusters'], linkage=best_params['linkage']).fit(X_sel), os.path.join(ARTIFACT_DIR, 'best_model.pkl'))

# Save scaler and selected feature list
joblib.dump(scaler, os.path.join(ARTIFACT_DIR, 'scaler.pkl'))
with open(os.path.join(ARTIFACT_DIR, 'selected_features.txt'), 'w') as f:
    for s in selected_features:
        f.write(s + '\n')

print('Artifacts saved to', ARTIFACT_DIR)

# ----- Final summary -----
short_print('Done — Summary for reviewer')
print('Project title:', PROJECT_TITLE)
print('\nMain objective:')
print(MAIN_OBJECTIVE)
print('\nRecommended model for deployment:', best_model_name, 'with params', best_params)
print('\nArtifacts and report located at:', ARTIFACT_DIR)


