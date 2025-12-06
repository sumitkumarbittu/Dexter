import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# 1. Create a sample DataFrame with numeric data
# We'll include some values that could be considered outliers
# to better demonstrate the differences between scalers.
data = {
    'lesson_id': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 500, 1500],
    'subject': ['Math', 'Science', 'History', 'Art', 'Math', 'Science', 'History', 'Art', 'Math', 'Science', 'Math', 'Science'],
    'description': [
        'Algebra Basics', 'Chemical Reactions', 'World War II', 'Impressionism',
        'Calculus I', 'Biology Basics', 'Ancient Rome', 'Sculpture Techniques',
        'Geometry Fundamentals', 'Physics I', 'Advanced Algebra', 'Quantum Mechanics'
    ]
}
df = pd.DataFrame(data)

print("Original DataFrame:")
print(df)
print("\nData types:")
print(df.dtypes)
print("-" * 50)

# Define the numeric column to be scaled
numeric_column = 'lesson_id'

print(f"Scaling '{numeric_column}' using different methods:\n")

# 2. Show different scaling methods

# --- StandardScaler ---
# 3. Explains when to use:
#    - Use when your data is approximately normally distributed (bell-shaped curve).
#    - It transforms data to have a mean of 0 and a standard deviation of 1.
#    - It's sensitive to outliers, as they can significantly impact the mean and standard deviation.
#    - Often used for algorithms that assume zero mean and unit variance (e.g., Linear Regression, Logistic Regression, SVMs, PCA).
print("1. StandardScaler (Z-score normalization):")
scaler_standard = StandardScaler()
df['lesson_id_standard_scaled'] = scaler_standard.fit_transform(df[[numeric_column]])
print(f"  Original Min: {df[numeric_column].min():.2f}, Max: {df[numeric_column].max():.2f}")
print(f"  Scaled Min: {df['lesson_id_standard_scaled'].min():.2f}, Max: {df['lesson_id_standard_scaled'].max():.2f}")
print(f"  Scaled Mean: {df['lesson_id_standard_scaled'].mean():.2f}, Std Dev: {df['lesson_id_standard_scaled'].std():.2f}")
print("  Explanation: It scales features to have 0 mean and unit variance. Suitable for normally distributed data, but sensitive to outliers.")
print("-" * 50)

# --- MinMaxScaler ---
# 3. Explains when to use:
#    - Use when you need features to be within a specific range, typically 0 to 1.
#    - It transforms data by shifting and scaling to fit within the specified range.
#    - It's also sensitive to outliers, as a single large outlier can compress the majority of data into a very small range.
#    - Useful for algorithms that don't assume any distribution of the data (e.g., K-Nearest Neighbors, Neural Networks (especially with sigmoid activation), image processing).
print("2. MinMaxScaler (Min-Max normalization):")
scaler_minmax = MinMaxScaler()
df['lesson_id_minmax_scaled'] = scaler_minmax.fit_transform(df[[numeric_column]])
print(f"  Original Min: {df[numeric_column].min():.2f}, Max: {df[numeric_column].max():.2f}")
print(f"  Scaled Min: {df['lesson_id_minmax_scaled'].min():.2f}, Max: {df['lesson_id_minmax_scaled'].max():.2f}")
print("  Explanation: It scales features to a fixed range (default 0 to 1). Preserves original distribution but sensitive to outliers.")
print("-" * 50)

# --- RobustScaler ---
# 3. Explains when to use:
#    - Use when your data contains many outliers, or when the data is not normally distributed.
#    - It scales features using statistics that are robust to outliers: it removes the median and scales the data according to the Interquartile Range (IQR).
#    - This makes it less sensitive to outliers compared to StandardScaler and MinMaxScaler.
#    - Ideal for data with skewness or heavy tails.
print("3. RobustScaler:")
scaler_robust = RobustScaler()
df['lesson_id_robust_scaled'] = scaler_robust.fit_transform(df[[numeric_column]])
print(f"  Original Min: {df[numeric_column].min():.2f}, Max: {df[numeric_column].max():.2f}")
print(f"  Scaled Min: {df['lesson_id_robust_scaled'].min():.2f}, Max: {df['lesson_id_robust_scaled'].max():.2f}")
print(f"  Scaled Median: {df['lesson_id_robust_scaled'].median():.2f}, IQR: {df['lesson_id_robust_scaled'].quantile(0.75) - df['lesson_id_robust_scaled'].quantile(0.25):.2f}")
print("  Explanation: It scales features using the median and Interquartile Range (IQR). Most suitable when your data contains outliers.")
print("-" * 50)

print("\nDataFrame with Scaled Features:")
print(df)

# Observe how the 'lesson_id' values like 500 and 1500 (our potential outliers)
# affect the range and distribution of scaled values differently across the methods.
# For example, in RobustScaler, the values 10-100 will be more tightly grouped around 0
# compared to StandardScaler or MinMaxScaler, because the median and IQR are less
# influenced by the extreme values.