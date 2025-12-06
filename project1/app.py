# remedial_recommender.py
# -----------------------------------------------
# Simulated ML model to recommend lessons based on student marks
# Workflow: Data Preprocessing → Weak Subject Detection → KNN → TF-IDF → Feedback
# -----------------------------------------------

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Stage 1: Data Preprocessing
# -----------------------------

# Simulated student marks (rows = students, cols = subjects)
data = {
    'Math': [80, 45, 60, 35, 90],
    'Science': [75, 40, 65, 50, 85],
    'English': [70, 55, 60, 45, 80],
    'History': [65, 30, 50, 40, 75]
}
students = ['S1', 'S2', 'S3', 'S4', 'S5']
df = pd.DataFrame(data, index=students)

# Normalize scores to 0–100 (in case input varies)
scaler = MinMaxScaler(feature_range=(0, 100))
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=students)

print("Normalized Student Scores:\n", df_scaled, "\n")

# -----------------------------
# Stage 2: Weak Subject Identification
# -----------------------------

threshold = 50  # below 50% = weak subject
weak_subjects = {}

for student in df_scaled.index:
    weak = df_scaled.columns[df_scaled.loc[student] < threshold].tolist()
    weak_subjects[student] = weak

print("Weak Subjects per Student:")
print(weak_subjects, "\n")

# -----------------------------
# Stage 3: Similarity-Based Recommendation (KNN)
# -----------------------------

# Represent each student as a vector of scores
X = df_scaled.values

# Use KNN to find peers with similar performance
k = 2
knn = NearestNeighbors(n_neighbors=k, metric='euclidean')
knn.fit(X)

# Find nearest neighbors for a target student (e.g., S2)
target_student = 'S2'
target_idx = students.index(target_student)
distances, indices = knn.kneighbors([X[target_idx]])

print(f"Nearest Neighbors for {target_student}:")
for i, idx in enumerate(indices[0]):
    if students[idx] != target_student:
        print(f"  {students[idx]} (distance={distances[0][i]:.2f})")
print()

# -----------------------------
# Stage 4: Lesson Tag Extraction (TF-IDF)
# -----------------------------

# Example lesson descriptions
lessons = {
    'Math': ["Algebra basics and number theory", 
              "Trigonometry and geometry practice"],
    'Science': ["Physics: motion and forces", 
                "Chemistry: atoms and compounds"],
    'English': ["Grammar and sentence structure", 
                "Essay writing techniques"],
    'History': ["World wars overview", 
                "Ancient civilizations study"]
}

# Flatten lessons for TF-IDF
lesson_texts = [desc for sublist in lessons.values() for desc in sublist]
subjects_flat = [sub for sub, sublist in lessons.items() for _ in sublist]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(lesson_texts)

# Suppose S2 is weak in Science and History
weak_tags = " ".join([sub.lower() for sub in weak_subjects[target_student]])
query_vec = vectorizer.transform([weak_tags])

# Compute similarity between weak topics and lessons
similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
top_indices = similarities.argsort()[::-1][:3]

print(f"Recommended Lessons for {target_student}:")
for idx in top_indices:
    print(f"  - {lesson_texts[idx]} (Score={similarities[idx]:.2f})")
print()

# -----------------------------
# Stage 5: Feedback Learning (Simulated)
# -----------------------------

# Simulate improvement after applying recommended lessons
before = np.mean(df_scaled.loc[target_student])
after = before + np.random.uniform(5, 10)  # simulate gain
improvement = after - before

print(f"Feedback Update for {target_student}:")
print(f"  Average score before: {before:.2f}")
print(f"  Average score after:  {after:.2f}")
print(f"  Improvement: +{improvement:.2f} points\n")

# Increase weight of effective lessons
if improvement > 5:
    print("✅ Lessons tagged as effective! Increasing recommendation weight for future use.")
else:
    print("⚠️ Minimal improvement — re-evaluate lesson effectiveness.")