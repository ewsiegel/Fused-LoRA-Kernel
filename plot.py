import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Read the file and clean up the column names
data = pd.read_csv('rslt.txt', sep='\s+', skiprows=3, names=['m', 'd', 'b', 'r', 'cublas_reference', 
                   'fused_sequential', 'fused_concurrent', 'fused_concurrent_asymmetric', 
                   'fused_concurrent_ax', 'Fastest_Impl'])

print(data)

# Filter for batch size 16 and r >= 16
data = data[(data['b'] == 16) & (data['r'] >= 16)]

# Calculate memory savings
#data['Memory Savings'] = 1 - (data['m'] * data['d'] + data['m'] * data['r'] + data['r'] * data['d']) / (data['m'] * data['d'] + data['d'] * data['b'])

# Calculate latency savings using the best performing implementation
#data['Latency Savings'] = 1 - 1/data[['fused_sequential', 'fused_concurrent', 'fused_concurrent_asymmetric', 'fused_concurrent_ax']].max(axis=1)

# Calculate total MACs for full rank matmul (in billions)
data['Total_MACs'] = data['m'].astype(float) * data['d'].astype(float) * data['b'].astype(float) / 1e9

# Calculate % runtime of reference (inverse of speedup × 100)
speedup = data[['fused_sequential', 'fused_concurrent', 'fused_concurrent_asymmetric', 'fused_concurrent_ax']].max(axis=1)
data['% Runtime of Reference'] = (1 / speedup) * 100

# Ensure mxd_size is a string
data['mxd_size'] = data['m'].astype(str) + 'x' + data['d'].astype(str)

# Calculate % memory usage of LoRA compared to full rank
data['% Memory Usage'] = ((data['m'] * data['r'] + data['r'] * data['d']) / (data['m'] * data['d'])) * 100

# Plot % memory usage vs m × d size
plt.figure(figsize=(12, 6))
for r in data['r'].unique():
    subset = data[data['r'] == r]
    color = sns.color_palette("husl")[3] if r == 64 else None  # Use theme blue for r=64
    plt.plot(subset['mxd_size'], subset['% Memory Usage'], label=f'r={r}', marker='o', color=color)

plt.xlabel('W Matrix Size (hidden_size × embedding_size)', fontsize=12)
plt.ylabel('LoRA Memory Savings\n(% Memory Usage vs Full Rank)', fontsize=12)
plt.title('LoRA Memory Savings vs Full Rank', fontsize=14)
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('memory_savings.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot % runtime of reference vs MACs
plt.figure(figsize=(12, 6))
for r in data['r'].unique():
    subset = data[data['r'] == r]
    color = sns.color_palette("husl")[3] if r == 64 else None
    plt.plot(subset['Total_MACs'], subset['% Runtime of Reference'], label=f'r={r}', marker='o', color=color)

# Add vertical dotted lines for specific m × d values
highlight_sizes = [8192, 4096, 2048, 1024]
for size in highlight_sizes:
    macs_value = (size * size * 16) / 1e9  # Calculate MACs for m = d = size, b = 16
    plt.axvline(x=macs_value, color='gray', linestyle='--', linewidth=1)
    plt.text(macs_value, plt.ylim()[1], f' {size}', rotation=0, verticalalignment='top', fontsize=10, color='gray')

plt.axhline(y=100, color='green', linestyle='--', linewidth=1.5)  # Add a dotted line at 100%
plt.xlabel('Total MACs in Full-Rank MatMul (billions)', fontsize=12)
plt.ylabel('Fused LoRA Kernel Latency Savings\n(% Runtime of Reference)', fontsize=12)
plt.title('Fused LoRA Kernel Latency Savings against Reference (CuBLAS torch backend) vs Computation Size (batch=16)\nembedding_size x hidden_size = {powers of 2 from 8192x8192 to 128x128}', fontsize=14)
plt.gca().invert_xaxis()
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('latency_savings.png', dpi=300, bbox_inches='tight')
plt.close()