import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Page settings
st.set_page_config(page_title="K-Fold Cross-Validation Simulator", layout="wide")
st.markdown("<style>body { background-color: #f9f9fa; }</style>", unsafe_allow_html=True)

# Header
st.title("🔁 K-Fold Cross-Validation")
st.markdown("""
This interactive tool shows **how K-Fold Cross-Validation** """)

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Simulation Settings")
    data_size = st.slider("📊 Total Data Points", 10, 100, 30, step=1)
    num_features = st.slider("🔢 Features per Sample", 2, 10, 5)
    k = st.slider("🔂 Number of Folds (K)", 2, min(10, data_size), 5)
    show_progress = st.checkbox("⏳ Show Fold Progress", value=True)

# Create dummy dataset
np.random.seed(42)
X = np.random.rand(data_size, num_features)
y = np.random.randint(0, 2, data_size)
data = np.arange(1, data_size + 1)

# Split data into folds
kf = KFold(n_splits=k)
folds = list(kf.split(X))

# Legend
# st.markdown("### 🎨 Color Legend")
legend_cols = st.columns([1, 1])
with legend_cols[0]:
    st.markdown("🟧 **Test Fold**")
with legend_cols[1]:
    st.markdown("🟦 **Train Folds**")

# Training loop
st.markdown("---")
st.subheader("🔍 Fold-by-Fold Visualization and Accuracy")

accuracies = []

for i, (train_idx, test_idx) in enumerate(folds):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Train model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    accuracies.append(acc)

    # Visualization
    fig, ax = plt.subplots(figsize=(12, 1.2))
    bar_colors = ['#ffa94d' if idx in test_idx else '#74c0fc' for idx in range(data_size)]
    ax.bar(range(data_size), np.ones(data_size), color=bar_colors, edgecolor='black')
    ax.set_xticks(range(data_size))
    ax.set_xticklabels(data, fontsize=8)
    ax.set_yticks([])
    ax.set_xlim(-1, data_size)
    ax.set_title(f"Fold {i+1} → Test indices: {list(data[test_idx])} | Accuracy: {acc:.2f}", 
                 fontsize=12, weight='bold', loc='left')
    test_patch = patches.Patch(color='#ffa94d', label='Test Fold')
    train_patch = patches.Patch(color='#74c0fc', label='Train Fold')
    ax.legend(handles=[train_patch, test_patch], loc='upper right')

    st.pyplot(fig)

    if show_progress:
        st.progress((i + 1) / k)
    st.markdown("---")

# Display overall results
st.success(f"✅ Mean Accuracy across {k} folds: **{np.mean(accuracies):.3f}**")
st.bar_chart(accuracies)

# Explanation
st.markdown("""
### ✅ Key Points Recap:
- Each fold acts as test data once, rest is used for training
- Accuracy is measured per fold and averaged
- Helps validate model generalization on different data splits
""")

# Footer
st.markdown("---")
st.caption("Created by Pavithra | Powered by Streamlit | YouTube: Pavithra’s Podcast 🎙️")