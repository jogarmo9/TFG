import sys; sys.path.insert(0,'scripts')
import pandas as pd, numpy as np, matplotlib, matplotlib.pyplot as plt
matplotlib.use('Agg')
import geo_utils as gu
from libpysal.weights import KNN as KNNWeights
from esda.moran import Moran

geo = pd.read_parquet('data/processed/predictions_geo.parquet')
geo['class_name'] = geo['class'].map(gu.CLASS_NAMES)

# Dynamic threshold per class: mu + sigma
thr = geo.groupby('class_name')['confidence'].transform(lambda x: x.mean() + x.std())
geo_filt = geo[geo.confidence >= thr].copy()
print(f'Total events after dynamic threshold: {len(geo_filt)}')
print(geo_filt.class_name.value_counts())

# Grid to ~200m cells
geo_filt['cell_lat'] = (geo_filt.lat / 0.0018).round() * 0.0018
geo_filt['cell_lon'] = (geo_filt.lon / 0.0022).round() * 0.0022

cells = geo_filt.groupby(['cell_lat','cell_lon']).size().reset_index(name='n_events')
print(f'\nGrid cells with events: {len(cells)}')

coords = cells[['cell_lat','cell_lon']].values
y = cells['n_events'].values.astype(float)

w = KNNWeights(coords, k=min(5, len(cells)-1))
w.transform = 'r'

mi = Moran(y, w, permutations=999)
print(f"\nMoran's I = {mi.I:.4f}, p-value = {mi.p_sim:.4f}, z = {mi.z_sim:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

z = (y - y.mean()) / y.std()
neighbors_list = [list(w.neighbors[i]) for i in range(len(z))]
weights_list   = [list(w.weights[i])   for i in range(len(z))]
lag_z = np.array([
    sum(weights_list[i][j] * z[neighbors_list[i][j]] for j in range(len(neighbors_list[i])))
    for i in range(len(z))
])

qcolors = []
for zi, lzi in zip(z, lag_z):
    if zi >= 0 and lzi >= 0:   qcolors.append('#c0392b')
    elif zi < 0 and lzi < 0:   qcolors.append('#2980b9')
    elif zi >= 0 and lzi < 0:  qcolors.append('#f39c12')
    else:                       qcolors.append('#27ae60')

axes[0].scatter(z, lag_z, c=qcolors, alpha=0.6, s=30, edgecolors='white', linewidths=0.3)
m, b = np.polyfit(z, lag_z, 1)
xline = np.linspace(z.min(), z.max(), 100)
axes[0].plot(xline, m*xline+b, 'k-', linewidth=1.5)
axes[0].axhline(0, color='gray', linewidth=0.5, linestyle='--')
axes[0].axvline(0, color='gray', linewidth=0.5, linestyle='--')
axes[0].set_xlabel('z(detecciones / celda)')
axes[0].set_ylabel('Media espacial del vecindario (lag)')
axes[0].set_title(f"Diagrama de dispersión de Moran\n$I = {mi.I:.3f}$,  $p = {mi.p_sim:.3f}$ (999 permutaciones)")

from matplotlib.patches import Patch
axes[0].legend(handles=[
    Patch(color='#c0392b', label='Alta-Alta (HH)'),
    Patch(color='#2980b9', label='Baja-Baja (LL)'),
    Patch(color='#f39c12', label='Alta-Baja (HL)'),
    Patch(color='#27ae60', label='Baja-Alta (LH)'),
], fontsize=8, loc='upper left')

axes[1].hist(mi.sim, bins=40, color='#95a5a6', edgecolor='white', alpha=0.8, label='Permutaciones')
axes[1].axvline(mi.I, color='#c0392b', linewidth=2, label=f'$I$ observado = {mi.I:.3f}')
axes[1].set_xlabel("Moran's I")
axes[1].set_title("Distribución bajo hipótesis nula (999 permutaciones)")
axes[1].legend(fontsize=9)

plt.suptitle("Autocorrelación espacial — todas las clases (umbral dinámico)", fontsize=12)
plt.tight_layout()
plt.savefig('outputs/moran_scatter.png', dpi=150, bbox_inches='tight')
print('Saved outputs/moran_scatter.png')
