import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load grouped metrics (change filename if needed)
df = pd.read_csv('adhd_grouped_model_metrics.csv')

# Filter for Screen Time factor
df_screen = df[df['Factor'].str.lower().str.contains('screen')]

# Print summary table for Screen Time
print('Model performance by Screen Time group:')
print(df_screen.to_markdown(index=False))

# Optional: Visualize R2 by Screen Time group and model
plt.figure(figsize=(10,6))
sns.barplot(data=df_screen, x='Group', y='R2', hue='Model')
plt.title('Model R² by Screen Time Group')
plt.ylabel('R²')
plt.xlabel('Screen Time Group')
plt.legend(title='Model')
plt.tight_layout()
plt.savefig('adhd_r2_by_screentime.png')
plt.show()

# Optional: If you want to see mean predicted ADHD risk by screen time group, you need access to the test set and model probabilities.
# Let me know if you want code for that as well.
