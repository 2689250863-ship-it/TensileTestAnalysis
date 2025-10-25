import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# --- Functions ---
def process_csv(file_name, width, thickness):
    df = pd.read_csv(file_name)
    df.columns = df.columns.str.strip()
    df['Stress'] = df['Force'] / (width * thickness)
    df['Strain'] = df['Strain'] / 100  # convert % to decimal
    df = df.groupby('Strain', as_index=False).mean().sort_values('Strain')
    return df

def calculate_properties(df, offset=0.002, linear_limit=0.002):
    # --- Young's modulus ---
    df_linear = df[df['Strain'] <= linear_limit]
    if len(df_linear) < 2:
        print("Error: Insufficient data points in linear region.")
        exit()

    X = df_linear['Strain'].values.reshape(-1, 1)
    y = df_linear['Stress'].values
    model = LinearRegression().fit(X, y)
    E = model.coef_[0]

    # --- Elastic limit ---
    linear_limit = 0.002  # 2% strain
    df_linear = df[df['Strain'] <= linear_limit]
    elastic_limit = df_linear['Stress'].max()
    elastic_strain = df_linear['Strain'][df_linear['Stress'].idxmax()]

    # --- Yield strength (0.2% offset) ---
    offset_line = E * (df['Strain'] - offset)
    idx = np.argmin(np.abs(offset_line - df['Stress']))
    yield_strength = df['Stress'].iloc[idx]
    strain_yield = df['Strain'].iloc[idx]

    # --- Ultimate tensile strength ---
    uts = df['Stress'].max()
    strain_uts = df['Strain'][df['Stress'].idxmax()]

    # --- Strain at failure ---
    strain_fail = df['Strain'].iloc[-1]
    stress_fail = df['Stress'].iloc[-1]

    return E, yield_strength, strain_yield, uts, strain_uts, strain_fail, stress_fail, elastic_limit, elastic_strain

# --- Material data ---
materials_data = {
    'PMMA': ("PMMA.csv", 12, 0.95),
    'Steel': ("Steel.csv", 12, 0.95),
    'ABS': ("ABS.csv", 12, 0.95),
    'AI6061': ("AI6061.csv", 12, 0.95)
}

# --- Process and calculate properties ---
materials = {}
for mat, (file, w, t) in materials_data.items():
    df = process_csv(file, w, t)
    props = calculate_properties(df)
    materials[mat] = (df, props)

    # --- Print Stress values and key properties ---
    E, ys, strain_yield, uts, strain_uts, strain_fail, stress_fail, elastic_limit, elastic_strain = props
    print(f"Material: {mat}")
    print("-" * 50)
    print("Stress values (MPa):")
    print("\nKey Properties:")
    print(f"  Young's Modulus: {E:.2f} MPa")
    print(f"  Elastic Limit: {elastic_limit:.2f} MPa at strain {elastic_strain*100:.2f}%")
    print(f"  Yield Strength (0.2% offset): {ys:.2f} MPa at strain {strain_yield*100:.2f}%")
    print(f"  Ultimate Tensile Strength: {uts:.2f} MPa at strain {strain_uts*100:.2f}%")
    print(f"  Strain at Failure: {strain_fail*100:.2f}% (Stress: {stress_fail:.2f} MPa)")
    print("-" * 50 + "\n")

# --- Plot curves in 2x2 subplots ---
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
axs = axs.flatten()
colors = {'PMMA':'blue', 'Steel':'red', 'ABS':'green', 'AI6061':'purple'}

for i, (mat, (df, props)) in enumerate(materials.items()):
    ax = axs[i]
    E, ys, strain_yield, uts, strain_uts, strain_fail, stress_fail, elastic_limit, elastic_strain = props

    # --- Plot original stress-strain curve ---
    ax.plot(df['Strain'], df['Stress'], color=colors[mat], linewidth=2, label=mat)

    # --- Linear fit (Young's modulus) limited to linear region ---
    x_linear = np.array([0, 0.003])  # 限制范围到弹性极限
    y_linear = E * x_linear
    ax.plot(x_linear, y_linear, color='black',linewidth=3, linestyle='--', label="Linear fit (E)")

    # Elastic Limit
    ax.scatter(elastic_strain, elastic_limit, color='orange', marker='D', s=60)
    ax.text(elastic_strain, elastic_limit, f'Elastic', fontsize=9, ha='right', va='center')

    # Yield Strength
    ax.scatter(strain_yield, ys, color='red', marker='o', s=60)
    ax.text(strain_yield+0.002, ys, f'Yield', fontsize=9, ha='left', va='center')

    # UTS
    ax.scatter(strain_uts, uts, color='purple', marker='^', s=60)
    ax.text(strain_uts, uts, f'UTS', fontsize=9, ha='right', va='center')

    # Strain at Failure
    ax.scatter(strain_fail, stress_fail, color='black', marker='s', s=60)
    ax.text(strain_fail+0.002, stress_fail, f'Failure', fontsize=9, ha='left', va='center')

    ax.set_xlabel('Tensile Strain')
    ax.set_ylabel('Stress (MPa)')
    ax.set_title(mat)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=8)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("Stress_Strain_4_Subplots_Properties.png", dpi=300)
plt.show()
