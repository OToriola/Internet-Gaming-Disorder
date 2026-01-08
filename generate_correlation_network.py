"""
Correlation Network Visualization for NSCH 2023
Creates beautiful network graph showing relationships between key variables
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_PATH = "data/nsch_2023e_topical.csv"
OUTPUT_DIR = "visualizations"

print("="*80)
print("CORRELATION NETWORK VISUALIZATION (NSCH 2023)")
print("="*80)

# Load data
print("\n1. Loading NSCH data...")
df = pd.read_csv(DATA_PATH)
print(f"   Loaded: {len(df):,} children")

# Extract key variables
print("\n2. Extracting key variables...")

variables = {
    'Screen Time': 'SCREENTIME',
    'Sleep Hours': 'HOURSLEEP',
    'Physical Activity': 'PHYSACTIV',
    'Depression/Anxiety': 'K2Q32A',
    'Behavioral Problems': 'K2Q33A',
    'ADHD': 'K2Q31A',
    'Age': 'SC_AGE_YEARS'
}

# Create clean dataframe with renamed columns
df_analysis = pd.DataFrame()
for label, col in variables.items():
    if col in df.columns:
        df_analysis[label] = pd.to_numeric(df[col], errors='coerce')

# For binary variables, convert to 0/1
binary_vars = ['Depression/Anxiety', 'Behavioral Problems', 'ADHD']
for var in binary_vars:
    if var in df_analysis.columns:
        # NSCH typically uses 1=Yes, 2=No, so recode
        df_analysis[var] = df_analysis[var].apply(lambda x: 1 if x == 1 else 0 if x == 2 else np.nan)

# Filter to valid data
df_analysis = df_analysis.dropna()
print(f"   Valid cases for analysis: {len(df_analysis):,}")

# Compute correlation matrix
print("\n3. Computing correlations...")
corr_matrix = df_analysis.corr()
print("\nCorrelation Matrix:")
print(corr_matrix.round(3))

# VISUALIZATION 1: Network Graph (Main Visualization)
print("\n4. Generating Visualization 1: Correlation network...")

fig, ax = plt.subplots(figsize=(40, 40))

# Create network graph
G = nx.Graph()

# Add nodes
for var in df_analysis.columns:
    G.add_node(var)

# Add edges for significant correlations
threshold = 0.10  # Show correlations above this threshold
for i, var1 in enumerate(df_analysis.columns):
    for j, var2 in enumerate(df_analysis.columns):
        if i < j:  # Only upper triangle
            corr = corr_matrix.loc[var1, var2]
            if abs(corr) >= threshold:
                G.add_edge(var1, var2, weight=abs(corr), correlation=corr)

# Set up positions using spring layout
pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

# Draw nodes
node_colors = []
node_categories = {
    'Screen Time': '#e74c3c',
    'Sleep Hours': '#9b59b6',
    'Physical Activity': '#2ecc71',
    'Depression/Anxiety': '#3498db',
    'Behavioral Problems': '#f39c12',
    'ADHD': '#1abc9c',
    'Age': '#34495e'
}

for node in G.nodes():
    node_colors.append(node_categories.get(node, '#95a5a6'))

nx.draw_networkx_nodes(
    G, pos,
    node_color=node_colors,
    node_size=12000,
    alpha=0.9,
    ax=ax
)

# Draw edges with varying thickness and color based on correlation
edges = G.edges()
weights = [G[u][v]['weight'] for u, v in edges]
correlations = [G[u][v]['correlation'] for u, v in edges]

# Positive correlations in green, negative in red
edge_colors = ['#2ecc71' if corr > 0 else '#e74c3c' for corr in correlations]

nx.draw_networkx_edges(
    G, pos,
    width=[w * 8 for w in weights],  # Scale edge width
    alpha=0.6,
    edge_color=edge_colors,
    ax=ax
)

# Draw labels
nx.draw_networkx_labels(
    G, pos,
    font_size=70,
    font_weight='bold',
    font_color='white',
    ax=ax
)

# Add edge labels showing correlation values
edge_labels = {(u, v): f"{G[u][v]['correlation']:.2f}" for u, v in G.edges()}
nx.draw_networkx_edge_labels(
    G, pos,
    edge_labels,
    font_size=52,
    font_weight='bold',
    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9),
    ax=ax
)

ax.set_title('Correlation Network: Screen Time, Sleep, Activity & Mental Health (NSCH 2023)\n'
             'Edge thickness = correlation strength | Green = positive, Red = negative',
             fontsize=70, fontweight='bold', pad=30)
ax.axis('off')

# Add legend with color explanations
legend_elements = [
    plt.scatter([], [], s=800, c='#e74c3c', label='Screen Time', edgecolors='black', linewidths=2),
    plt.scatter([], [], s=800, c='#9b59b6', label='Sleep Hours', edgecolors='black', linewidths=2),
    plt.scatter([], [], s=800, c='#2ecc71', label='Physical Activity', edgecolors='black', linewidths=2),
    plt.scatter([], [], s=800, c='#3498db', label='Depression/Anxiety', edgecolors='black', linewidths=2),
    plt.scatter([], [], s=800, c='#f39c12', label='Behavioral Problems', edgecolors='black', linewidths=2),
    plt.scatter([], [], s=800, c='#1abc9c', label='ADHD', edgecolors='black', linewidths=2),
    plt.scatter([], [], s=800, c='#34495e', label='Age', edgecolors='black', linewidths=2)
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=48, framealpha=0.9, 
          title='Variable Categories', title_fontsize=52)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/correlation_network.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"   [OK] Saved: {OUTPUT_DIR}/correlation_network.png")

# VISUALIZATION 2: Circular Correlation Network
print("\n5. Generating Visualization 2: Circular correlation network...")

fig, ax = plt.subplots(figsize=(40, 40))

# Use circular layout
pos_circular = nx.circular_layout(G)

# Draw with circular layout
nx.draw_networkx_nodes(
    G, pos_circular,
    node_color=node_colors,
    node_size=12000,
    alpha=0.9,
    ax=ax
)

nx.draw_networkx_edges(
    G, pos_circular,
    width=[w * 12 for w in weights],
    alpha=0.5,
    edge_color=edge_colors,
    ax=ax
)

nx.draw_networkx_labels(
    G, pos_circular,
    font_size=70,
    font_weight='bold',
    font_color='white',
    ax=ax
)

# Add correlation values
edge_labels = {(u, v): f"{G[u][v]['correlation']:.2f}" for u, v in G.edges()}
nx.draw_networkx_edge_labels(
    G, pos_circular,
    edge_labels,
    font_size=52,
    font_weight='bold',
    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9),
    ax=ax
)

ax.set_title('Circular Correlation Network (NSCH 2023)\n'
             f'Showing correlations ≥ {threshold} | Green = positive, Red = negative',
             fontsize=70, fontweight='bold', pad=30)
ax.axis('off')
ax.legend(handles=legend_elements, loc='upper left', fontsize=48, framealpha=0.9,
          title='Variable Categories', title_fontsize=52)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/correlation_network_circular.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"   [OK] Saved: {OUTPUT_DIR}/correlation_network_circular.png")

# VISUALIZATION 3: Heatmap + Network Combo
print("\n6. Generating Visualization 3: Combined heatmap and network...")

fig = plt.figure(figsize=(64, 30))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2])

# Left: Enhanced correlation heatmap
ax1 = fig.add_subplot(gs[0, 0])

# Create mask for upper triangle
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    fmt='.2f',
    cmap='RdYlGn',
    center=0,
    square=True,
    linewidths=4,
    cbar_kws={'label': 'Correlation Coefficient', 'shrink': 0.8},
    annot_kws={'fontsize': 50, 'fontweight': 'bold'},
    ax=ax1,
    vmin=-0.5,
    vmax=0.5
)

ax1.set_title('Correlation Heatmap', fontsize=70, fontweight='bold', pad=40)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=60)
ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0, fontsize=60)

# Right: Network graph
ax2 = fig.add_subplot(gs[0, 1])

# Draw network
nx.draw_networkx_nodes(
    G, pos,
    node_color=node_colors,
    node_size=14000,
    alpha=0.9,
    ax=ax2
)

nx.draw_networkx_edges(
    G, pos,
    width=[w * 14 for w in weights],
    alpha=0.6,
    edge_color=edge_colors,
    ax=ax2
)

nx.draw_networkx_labels(
    G, pos,
    font_size=72,
    font_weight='bold',
    font_color='white',
    ax=ax2
)

edge_labels = {(u, v): f"{G[u][v]['correlation']:.2f}" for u, v in G.edges()}
nx.draw_networkx_edge_labels(
    G, pos,
    edge_labels,
    font_size=52,
    font_weight='bold',
    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9),
    ax=ax2
)

ax2.set_title('Network Visualization', fontsize=70, fontweight='bold', pad=40)
ax2.axis('off')

plt.suptitle('Variable Relationships: Correlation Matrix & Network (NSCH 2023)', 
             fontsize=75, fontweight='bold', y=0.98)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/correlation_heatmap_network_combo.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"   [OK] Saved: {OUTPUT_DIR}/correlation_heatmap_network_combo.png")

# Summary statistics
print("\n" + "="*80)
print("CORRELATION NETWORK ANALYSIS COMPLETE - KEY FINDINGS:")
print("="*80)

print("\nStrongest Positive Correlations:")
corr_values = []
for i, var1 in enumerate(df_analysis.columns):
    for j, var2 in enumerate(df_analysis.columns):
        if i < j:
            corr_values.append((var1, var2, corr_matrix.loc[var1, var2]))

corr_values.sort(key=lambda x: x[2], reverse=True)
for var1, var2, corr in corr_values[:5]:
    print(f"  - {var1} ↔ {var2}: r = {corr:.3f}")

print("\nStrongest Negative Correlations:")
corr_values.sort(key=lambda x: x[2])
for var1, var2, corr in corr_values[:5]:
    print(f"  - {var1} ↔ {var2}: r = {corr:.3f}")

print("\nNetwork Statistics:")
print(f"  - Number of nodes: {G.number_of_nodes()}")
print(f"  - Number of edges: {G.number_of_edges()}")
print(f"  - Network density: {nx.density(G):.3f}")

print("\n" + "="*80)
print("Generated 3 correlation network visualizations:")
print("  1. correlation_network.png (spring layout)")
print("  2. correlation_network_circular.png (circular layout)")
print("  3. correlation_heatmap_network_combo.png (combined view)")
print("="*80)
