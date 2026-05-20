import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# LOAD DATASET
iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

df["species"] = iris.target

df["species"] = df["species"].map({
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
})

# TITLE
st.title("Iris Species Classification")

st.write("Machine Learning Dashboard using Streamlit")

# FEATURES AND TARGET
X = iris.data
y = iris.target

# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# MODEL
model = RandomForestClassifier()

model.fit(X_train, y_train)

# PREDICTIONS
y_pred = model.predict(X_test)

# METRICS
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

# SHOW METRICS
st.header("Model Metrics")

st.write(f"Accuracy: {accuracy:.2f}")
st.write(f"Precision: {precision:.2f}")
st.write(f"Recall: {recall:.2f}")
st.write(f"F1 Score: {f1:.2f}")

# USER INPUT
st.header("Predict a Flower")

sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.0)
sepal_width = st.slider("Sepal Width", 2.0, 5.0, 3.0)
petal_length = st.slider("Petal Length", 1.0, 7.0, 4.0)
petal_width = st.slider("Petal Width", 0.1, 3.0, 1.0)

sample = [[
    sepal_length,
    sepal_width,
    petal_length,
    petal_width
]]

prediction = model.predict(sample)

species_names = {
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
}

predicted_species = species_names[prediction[0]]

st.success(f"Predicted Species: {predicted_species}")

# 3D PLOT
st.header("3D Visualization")

fig = px.scatter_3d(
    df,
    x='sepal length (cm)',
    y='sepal width (cm)',
    z='petal length (cm)',
    color='species'
)

fig.add_scatter3d(
    x=[sepal_length],
    y=[sepal_width],
    z=[petal_length],
    mode='markers',
    marker=dict(size=10, color='black'),
    name='New Sample'
)

st.plotly_chart(fig)

# HISTOGRAM
st.header("Histogram")

hist = px.histogram(
    df,
    x='petal length (cm)',
    color='species',
    barmode='overlay'
)

st.plotly_chart(hist)
