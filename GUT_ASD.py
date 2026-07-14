import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# ── Load ──────────────────────────────────────────────
df = pd.read_csv('E:\mini_thesis\ASD.csv')

# ── Transpose: rows=samples, cols=bacteria ────────────
taxonomy = df['Taxonomy'].values
data = df.drop('Taxonomy', axis=1).T
data.columns = taxonomy
data.index.name = 'Sample'

# ── Labels ────────────────────────────────────────────
data['Label'] = ['ASD' if s.startswith('A') else 'Control' 
                 for s in data.index]

print(data['Label'].value_counts())
print(data.shape)

# ── Relative Abundance Normalization ──────────────────
X = data.drop('Label', axis=1)
X = X.div(X.sum(axis=1), axis=0)  

y = data['Label']

# ── Feature Selection: HIGHEST variance ──────────────
variances = X.var()
top_features = variances.nlargest(100).index
X = X[top_features]

print("Features after selection:", X.shape[1])

# ── Model با Cross-Validation ─────────────────────────
model = RandomForestClassifier(n_estimators=100, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

print(f"\nCV Accuracy: {scores.mean():.2f} ± {scores.std():.2f}")

# ── Train Feature Importance ────────
model.fit(X, y)

# ── Top Bacteria ──────────────────────────────────────
importances = pd.Series(model.feature_importances_, index=X.columns)
top20 = importances.nlargest(20)

plt.figure(figsize=(10, 6))
top20.sort_values().plot(kind='barh', color='#4fc3f7')
plt.title('Top 20 Most Important Bacteria (ASD vs Control)')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('bacteria_importance.png')
plt.show()

# ── Confusion Matrix ──────────────────────────────────
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model2 = RandomForestClassifier(n_estimators=100, random_state=42)
model2.fit(X_train, y_train)
y_pred = model2.predict(X_test)

print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=['ASD', 'Control'])
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=['ASD', 'Control'],
            yticklabels=['ASD', 'Control'],
            cmap='Blues')
plt.title('Confusion Matrix — ASD vs Control')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix_asd.png')
plt.show()