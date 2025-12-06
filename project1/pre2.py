"""
This Python code demonstrates various methods for encoding categorical variables using `pandas` and `sklearn`,
along with explanations of when to use each technique.
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np

# 1. Create a sample dataframe with categorical data
print("1. Creating Sample DataFrame\n")

data = {
    'lesson_id': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115],
    'subject': ['Math', 'Science', 'History', 'Math', 'Art', 'Science', 'Math', 'History', 'Art', 'Science', 'English', 'Math', 'Science', 'English', 'History'],
    'description': ['Algebra Basics', 'Chemical Reactions', 'World War II', 'Geometry', 'Impressionism',
                    'Ecology', 'Calculus', 'Ancient Civilizations', 'Sculpture', 'Physics Basics',
                    'Shakespeare', 'Trigonometry', 'Biology', 'Grammar', 'Modern History'],
    'student_score': [85, 92, 78, 90, 65, 88, 95, 80, 70, 91, 75, 89, 93, 72, 81] # Added a numerical target for Target Encoding
}
df = pd.DataFrame(data)

print("Original DataFrame:")
print(df)
print("\nData Types:")
print(df.dtypes)

# Identify categorical columns
categorical_cols = ['subject', 'description']

print("\n" + "="*70)
print("             2. Demonstrating Different Encoding Methods")
print("="*70)

# --- Encoding Method 1: Label Encoding ---
print("\n--- Label Encoding ---")

# Create a copy to avoid modifying the original DataFrame
df_label_encoded = df.copy()

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Apply Label Encoding to the 'subject' column
df_label_encoded['subject_encoded'] = label_encoder.fit_transform(df_label_encoded['subject'])

print("\nDataFrame after Label Encoding 'subject' column (first 5 rows):")
print(df_label_encoded[['subject', 'subject_encoded']].head())

print(f"\nOriginal 'subject' categories: {label_encoder.classes_}")
print(f"Encoded 'subject' values: {df_label_encoded['subject_encoded'].unique()}")


# --- Explanation: Label Encoding ---
print("\n" + "="*70)
print("3. Explanation: Label Encoding")
print("="*70)
print("What it does: Assigns a unique integer to each category based on alphabetical order (0, 1, 2, ...).")
print("Pros:")
print("  - Simple and memory efficient, as it adds only one column.")
print("  - Useful for tree-based algorithms (Decision Trees, Random Forests, Gradient Boosting) that can implicitly handle ordinal relationships.")
print("Cons:")
print("  - Introduces an artificial ordinal relationship between categories. For example, 'Art' (0) < 'History' (1) < 'Math' (2) implies an order that does not exist for nominal data.")
print("  - Can mislead linear models (e.g., Linear Regression, SVM) which might interpret higher numbers as higher importance or magnitude.")
print("When to use:")
print("  - For **ordinal categorical data** where the order of categories truly matters (e.g., 'small', 'medium', 'large' can be encoded as 0, 1, 2).")
print("  - When using **tree-based machine learning models**, as they are less sensitive to the artificial order.")
print("  - As a quick-and-dirty approach when memory is a concern, but be mindful of the introduced ordinality for non-ordinal data.")


# --- Encoding Method 2: One-Hot Encoding ---
print("\n--- One-Hot Encoding ---")

# Method 1: Using pandas get_dummies (simpler for direct DataFrame manipulation)
# Create new binary columns for each unique category.
# 'prefix' adds a prefix to the new column names (e.g., 'sub_Math', 'desc_Algebra Basics').
# 'dtype=int' ensures the new columns are integers (0 or 1) instead of booleans.
df_onehot_encoded_pd = pd.get_dummies(df, columns=categorical_cols, prefix=['sub', 'desc'], dtype=int)

print("\nDataFrame after One-Hot Encoding 'subject' and 'description' (using pd.get_dummies, first 5 rows):")
print(df_onehot_encoded_pd.head())
print(f"\nOriginal 'subject' column has been replaced by {len(df['subject'].unique())} new columns (e.g., sub_Art, sub_English, sub_History, etc.).")
print(f"Original 'description' column has been replaced by {len(df['description'].unique())} new columns.")


# Method 2: Using sklearn.preprocessing.OneHotEncoder (more suitable for pipelines and sparse output)
print("\n--- Using sklearn.preprocessing.OneHotEncoder (for pipeline integration) ---")
# Select only the categorical columns for sklearn encoder
data_for_ohe_sklearn = df[categorical_cols]

# Initialize OneHotEncoder
# handle_unknown='ignore': Allows encoding of unseen categories as all zeros during transform (useful in production).
# sparse_output=False: Returns a dense NumPy array instead of a sparse matrix (easier to view here).
ohe_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# Fit and transform the selected columns
ohe_features = ohe_encoder.fit_transform(data_for_ohe_sklearn)

# Get feature names for the new columns
ohe_feature_names = ohe_encoder.get_feature_names_out(categorical_cols)

# Create a DataFrame from the one-hot encoded features
df_ohe_sklearn = pd.DataFrame(ohe_features, columns=ohe_feature_names, index=df.index)

# Concatenate with the original DataFrame (dropping original categorical columns)
df_combined_ohe_sklearn = pd.concat([df.drop(columns=categorical_cols), df_ohe_sklearn], axis=1)

print("\nDataFrame after One-Hot Encoding with sklearn.OneHotEncoder (first 5 rows):")
print(df_combined_ohe_sklearn.head())


# --- Explanation: One-Hot Encoding ---
print("\n" + "="*70)
print("3. Explanation: One-Hot Encoding")
print("="*70)
print("What it does: Creates new binary columns (dummy variables) for each unique category present in the original column. If a row belongs to a category, its corresponding dummy variable is 1, otherwise 0.")
print("Pros:")
print("  - Does not introduce artificial ordinal relationships, making it suitable for **nominal data**.")
print("  - Preferred for linear models (Logistic Regression, SVM, Linear Regression) and neural networks, as they assume no inherent order.")
print("  - Easy to interpret the impact of each category on the model.")
print("Cons:")
print("  - Can lead to a high-dimensional sparse dataset, especially with high cardinality categorical variables (many unique categories). This is known as the 'curse of dimensionality'.")
print("  - Increased memory consumption and computational cost.")
print("  - Can suffer from multicollinearity if all dummy variables are included (dummy variable trap). Typically, one column is dropped per original categorical variable (e.g., using `drop_first=True` in `pd.get_dummies` or `drop='first'` in `OneHotEncoder`).")
print("When to use:")
print("  - For **nominal categorical data** where there is no inherent order (e.g., 'Math', 'Science', 'Art').")
print("  - When using **linear models, neural networks, or any model sensitive to artificial ordinality**.")
print("  - When the number of unique categories is relatively small to moderate.")


# --- Encoding Method 3: Target Encoding ---
print("\n--- Target Encoding ---")

# Create a copy for target encoding
df_target_encoded = df.copy()

# For Target Encoding, we need a numerical target variable. Here, 'student_score' is our target.
# We will encode 'subject' based on the mean 'student_score' for each subject.
# IMPORTANT: For real-world applications, this must be done carefully using cross-validation or smoothing
# techniques to prevent data leakage and overfitting. This example shows a basic implementation.

# Calculate the mean 'student_score' for each 'subject'
target_mean_map_subject = df_target_encoded.groupby('subject')['student_score'].mean().to_dict()

# Create a new column 'subject_target_encoded' by mapping these means
df_target_encoded['subject_target_encoded'] = df_target_encoded['subject'].map(target_mean_map_subject)

print("\nDataFrame after Target Encoding 'subject' column (first 5 rows):")
print(df_target_encoded[['subject', 'student_score', 'subject_target_encoded']].head())

print("\nCalculated Mean 'student_score' by 'subject':")
print(pd.Series(target_mean_map_subject))

# --- Explanation: Target Encoding ---
print("\n" + "="*70)
print("3. Explanation: Target Encoding")
print("="*70)
print("What it does: Replaces each category with the mean (or other aggregate statistic) of the target variable for that category. For example, if 'Math' classes have an average 'student_score' of 89.66, all 'Math' entries are replaced with 89.66.")
print("Pros:")
print("  - Very effective for **high cardinality categorical variables**, as it doesn't increase dimensionality (adds only one new feature).")
print("  - Captures information about the target variable directly into the feature, often leading to better model performance.")
print("  - Reduces the number of features significantly compared to One-Hot Encoding for high cardinality variables.")
print("Cons:")
print("  - **Highly prone to overfitting and data leakage** if not implemented carefully (e.g., calculating target means on the full dataset and then applying to the test set). If a category has very few samples, its mean can be noisy.")
print("  - Requires a numerical target variable, so it's a **supervised encoding method**.")
print("When to use:")
print("  - For **high cardinality categorical variables** where One-Hot Encoding would create too many features.")
print("  - When you have a numerical target variable and want to capture the relationship between categories and the target.")
print("  - **Always use with caution**: Implement techniques like K-fold target encoding, leave-one-out encoding, or additive smoothing to prevent overfitting and data leakage. Libraries like `category_encoders` provide robust implementations.")
print("\n" + "="*70)
print("                        SUMMARY OF ENCODING METHODS")
print("="*70)
print("""
1. Label Encoding:
   - How it works: Assigns a unique integer (0, 1, 2, ...) to each category.
   - Best for: Ordinal data (e.g., 'low', 'medium', 'high') or tree-based models.
   - Avoid for: Nominal data with linear models (introduces artificial order).

2. One-Hot Encoding:
   - How it works: Creates a new binary column for each unique category.
   - Best for: Nominal data and models sensitive to numerical magnitude.
   - Avoid for: High-cardinality features (causes high dimensionality).

3. Target Encoding:
   - How it works: Replaces categories with target variable statistics.
   - Best for: High-cardinality features with strong target relationship.
   - Warning: Prone to overfitting; requires careful implementation.
""")