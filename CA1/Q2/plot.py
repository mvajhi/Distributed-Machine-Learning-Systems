import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# --- Chart 1: Global Accuracy Comparison ---
# Data derived from scenario_1.out, scenario_2.out, and malicious.out

accuracy_data = {
    'Scenario': ['Scenario 1 (1R, 10E)', 'Scenario 2 (10R, 1E)', 'Scenario 3 (Malicious)'],
    'Global Accuracy': [0.6277, 0.6358, 0.5762]
}
df_accuracy = pd.DataFrame(accuracy_data)
# Sort by accuracy
df_accuracy = df_accuracy.sort_values(by='Global Accuracy', ascending=False)

plt.figure(figsize=(10, 6))
bars = plt.bar(df_accuracy['Scenario'], df_accuracy['Global Accuracy'], color=['blue', 'green', 'red'])
plt.ylabel('Global Average Accuracy')
plt.title('Global Model Accuracy Comparison')
plt.ylim(0.5, 0.7) # Adjust ylim to better show differences

# Add data labels
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.005, f'{yval:.4f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('q2_accuracy_comparison.png')
print("Saved q2_accuracy_comparison.png")


# --- Chart 2: Total Execution Time Comparison ---
# Data derived from Rank 0 in scenario_1.out, scenario_2.out, and malicious.out

time_data = {
    'Scenario': ['Scenario 1 (1R, 10E)', 'Scenario 2 (10R, 1E)', 'Scenario 3 (Malicious)'],
    'Total Execution Time (s)': [15.22, 17.06, 17.39] # Time from Rank 0
}
df_time = pd.DataFrame(time_data)
# Sort by time
df_time = df_time.sort_values(by='Total Execution Time (s)', ascending=True)

plt.figure(figsize=(10, 6))
bars_time = plt.bar(df_time['Scenario'], df_time['Total Execution Time (s)'], color=['cyan', 'orange', 'red'])
plt.ylabel('Total Execution Time (seconds)')
plt.title('Total Execution Time (Rank 0)')

# Add data labels
for bar in bars_time:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, f'{yval:.2f} s', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('q2_time_comparison.png')
print("Saved q2_time_comparison.png")


# --- Chart 3: Malicious Scenario - Local vs Global Accuracy ---
# Data derived from malicious.out

malicious_acc_data = {
    'Entity': ['Global Model', 'Client 1 (Malicious)', 'Client 2 (Normal)', 'Client 3 (Normal)'],
    'Accuracy': [0.5762, 0.6957, 0.4662, 0.5667]
}
df_malicious_acc = pd.DataFrame(malicious_acc_data)
# Sort by accuracy
df_malicious_acc = df_malicious_acc.sort_values(by='Accuracy', ascending=False)

colors = ['red' if 'Global' in x else ('orange' if 'Malicious' in x else 'gray') for x in df_malicious_acc['Entity']]

plt.figure(figsize=(12, 7))
bars_malicious = plt.bar(df_malicious_acc['Entity'], df_malicious_acc['Accuracy'], color=colors)
plt.ylabel('Final Accuracy')
plt.title('Malicious Scenario: Final Local vs. Global Accuracy')
plt.ylim(0.4, 0.75)

# Add data labels
for bar in bars_malicious:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.005, f'{yval:.4f}', ha='center', va='bottom')

# Create custom legend
legend_patches = [
    mpatches.Patch(color='red', label='Global Model Accuracy'),
    mpatches.Patch(color='orange', label='Malicious Client (Rank 1)'),
    mpatches.Patch(color='gray', label='Normal Clients (Rank 2 & 3)')
]
plt.legend(handles=legend_patches)

plt.tight_layout()
plt.savefig('q2_malicious_accuracy.png')
print("Saved q2_malicious_accuracy.png")