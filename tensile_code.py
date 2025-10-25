import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# --- Functions ---
def process_csv(file_name, width, thickness):
    df = pd.read_csv(file_name)
    df.columns = df.columns.str.strip()
    df['Stress'] = df['Force'] / (width * thickness)
    df['Strain'] = df['Strain'] / 100
    df = df.groupby('Strain', as_index=False).mean().sort_values('Strain')
    return df

def calculate_properties(df, offset=0.002, linear_limit=0.002):
    df_linear = df[df['Strain'] <= linear_limit]
    X = df_linear['Strain'].values.reshape(-1, 1)
    y = df_linear['Stress'].values
    E = LinearRegression().fit(X, y).coef_[0]

    elastic_limit = df_linear['Stress'].max()
    elastic_strain = df_linear['Strain'][df_linear['Stress'].idxmax()]

    offset_line = E * (df['Strain'] - offset)
    idx = np.argmin(np.abs(offset_line - df['Stress']))
    yield_strength = df['Stress'].iloc[idx]
    strain_yield = df['Strain'].iloc[idx]

    uts = df['Stress'].max()
    strain_uts = df['Strain'][df['Stress'].idxmax()]

    strain_fail = df['Strain'].iloc[-1]
    stress_fail = df['Stress'].iloc[-1]

    return E, yield_strength, strain_yield, uts, strain_uts, strain_fail, stress_fail, elastic_limit, elastic_strain

# --- Material data ---
PMMA_file = "PMMA.csv"
Steel_file = "Steel.csv"
ABS_file = "ABS.csv"
AI6061_file = "AI6061.csv"

PMMA_width, PMMA_thickness = 12, 0.95
Steel_width, Steel_thickness = 12, 0.95
ABS_width, ABS_thickness = 12, 0.95
AI6061_width, AI6061_thickness = 12, 0.95

colors = {'PMMA':'blue', 'Steel':'red', 'ABS':'green', 'AI6061':'purple'}

# --- Process data ---
PMMA_df = process_csv(PMMA_file, PMMA_width, PMMA_thickness)
Steel_df = process_csv(Steel_file, Steel_width, Steel_thickness)
ABS_df = process_csv(ABS_file, ABS_width, ABS_thickness)
AI6061_df = process_csv(AI6061_file, AI6061_width, AI6061_thickness)

PMMA_props = calculate_properties(PMMA_df)
Steel_props = calculate_properties(Steel_df)
ABS_props = calculate_properties(ABS_df)
AI6061_props = calculate_properties(AI6061_df)

# --- Create 2x2 subplot ---
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# --- PMMA ---
E, ys, strain_yield, uts, strain_uts, strain_fail, stress_fail, elastic_limit, elastic_strain = PMMA_props
ax = axs[0,0]
ax.plot(PMMA_df['Strain'], PMMA_df['Stress'], color=colors['PMMA'], linewidth=2, label='PMMA Stress-Strain')
ax.plot([0,0.003], [0,E*0.003], color='black', linestyle='--', linewidth=2, label='Linear fit (E)')
ax.scatter(strain_uts, uts, color='purple', marker='^', s=60)
ax.text(strain_uts, uts, 'UTS', fontsize=9, ha='right')
ax.scatter(strain_fail, stress_fail, color='black', marker='s', s=60)
ax.text(strain_fail+0.002, stress_fail, 'Failure', fontsize=9, ha='left')
ax.set_xlabel('Tensile Strain')
ax.set_ylabel('Stress (MPa)')
ax.set_title('PMMA')
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(fontsize=8)

# --- Steel ---
E, ys, strain_yield, uts, strain_uts, strain_fail, stress_fail, elastic_limit, elastic_strain = Steel_props
ax = axs[0,1]
ax.plot(Steel_df['Strain'], Steel_df['Stress'], color=colors['Steel'], linewidth=2, label='Steel Stress-Strain')
ax.plot([0,0.003], [0,E*0.003], color='black', linestyle='--', linewidth=2, label='Linear fit (E)')
ax.scatter(elastic_strain, elastic_limit, color='orange', marker='D', s=60)
ax.text(elastic_strain, elastic_limit, 'Elastic', fontsize=9, ha='right')
ax.scatter(strain_yield, ys, color='red', marker='o', s=60)
ax.text(strain_yield+0.002, ys, 'Yield', fontsize=9, ha='left')
ax.scatter(strain_uts, uts, color='purple', marker='^', s=60)
ax.text(strain_uts, uts, 'UTS', fontsize=9, ha='right')
ax.scatter(strain_fail, stress_fail, color='black', marker='s', s=60)
ax.text(strain_fail+0.002, stress_fail, 'Failure', fontsize=9, ha='left')
ax.set_xlabel('Tensile Strain')
ax.set_ylabel('Stress (MPa)')
ax.set_title('Steel')
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(fontsize=8)

# --- ABS ---
E, ys, strain_yield, uts, strain_uts, strain_fail, stress_fail, elastic_limit, elastic_strain = ABS_props
ax = axs[1,0]
ax.plot(ABS_df['Strain'], ABS_df['Stress'], color=colors['ABS'], linewidth=2, label='ABS Stress-Strain')
ax.plot([0,0.003], [0,E*0.003], color='black', linestyle='--', linewidth=2, label='Linear fit (E)')
ax.scatter(elastic_strain+0.015, elastic_limit+95, color='orange', marker='D', s=60)
ax.text(elastic_strain+0.015, elastic_limit+95, 'Elastic', fontsize=9, ha='right')
ax.scatter(strain_uts, uts, color='purple', marker='^', s=60)
ax.text(strain_uts, uts, 'UTS/Yield strength', fontsize=9, ha='right')
ax.scatter(strain_fail, stress_fail, color='black', marker='s', s=60)
ax.text(strain_fail+0.002, stress_fail, 'Failure', fontsize=9, ha='left')
ax.set_xlabel('Tensile Strain')
ax.set_ylabel('Stress (MPa)')
ax.set_title('ABS')
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(fontsize=8)

# --- AI6061 ---
E, ys, strain_yield, uts, strain_uts, strain_fail, stress_fail, elastic_limit, elastic_strain = AI6061_props
ax = axs[1,1]
ax.plot(AI6061_df['Strain'], AI6061_df['Stress'], color=colors['AI6061'], linewidth=2, label='AI6061 Stress-Strain')
ax.plot([0,0.003], [0,E*0.003], color='black', linestyle='--', linewidth=2, label='Linear fit (E)')
ax.scatter(elastic_strain, elastic_limit, color='orange', marker='D', s=60)
ax.text(elastic_strain, elastic_limit, 'Elastic', fontsize=9, ha='right')
ax.scatter(strain_yield, ys, color='red', marker='o', s=60)
ax.text(strain_yield+0.002, ys, 'Yield', fontsize=9, ha='left')
ax.scatter(strain_uts, uts, color='purple', marker='^', s=60)
ax.text(strain_uts, uts, 'UTS', fontsize=9, ha='right')
ax.scatter(strain_fail, stress_fail, color='black', marker='s', s=60)
ax.text(strain_fail+0.002, stress_fail, 'Failure', fontsize=9, ha='left')
ax.set_xlabel('Tensile Strain')
ax.set_ylabel('Stress (MPa)')
ax.set_title('AI6061')
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(fontsize=8)

plt.tight_layout(rect=[0,0.03,1,0.95])
plt.savefig("Stress_Strain_4_Subplots_SeparatePoints.png", dpi=300)
plt.show()
