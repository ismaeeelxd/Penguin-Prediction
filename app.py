import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from preprocessing.transforms import encode_categoricals, fill_categorical_missing, fill_numeric_missing, standardize_train_test
from utils.utils import generate_confusion_matrix, load_config
from data.loader import load_dataset, get_feature_target


def main():
    st.title("Penguin Species Classifier")

    config = load_config()

    st.sidebar.header("Configuration")

    model_name = st.sidebar.radio(
        "Select Model",
        list(config.MODELS.keys()),
        index=list(config.MODELS.keys()).index(config.SELECTED_MODEL)
    )


    config.SELECTED_MODEL = model_name
    model_class = config.MODELS[model_name]

    class_names = list(config.TARGET_CLASSES.keys())

    selected_idx1 = 0
    selected_idx2 = 1

    c1 = st.sidebar.selectbox("Select Class 1", class_names, index=selected_idx1)
    c2 = st.sidebar.selectbox("Select Class 2", class_names, index=selected_idx2)

    config.SELECTED_CLASSES = [c1, c2]
    all_features = list(config.FEATURES_COLUMNS)
    f1 = st.sidebar.selectbox("Feature 1", all_features, index=0)
    f2 = st.sidebar.selectbox("Feature 2", all_features, index=1)
    config.SELECTED_FEATURES = [f1, f2]

    config.LEARNING_RATE = st.sidebar.number_input("Learning Rate", 0.0001, 1.0, config.LEARNING_RATE, step=0.01)
    config.N_ITERS = st.sidebar.number_input("Epochs", 1, 100000, config.N_ITERS, step=1)
    config.MSE_THRESHOLD = st.sidebar.number_input("MSE Threshold", 0.0, 1.0, config.MSE_THRESHOLD, step=0.001)
    config.BIAS = st.sidebar.checkbox("Add Bias", value=config.BIAS)
    config.TEST_SIZE = st.sidebar.slider("Test Size", 0.1, 0.9, config.TEST_SIZE)

    if st.sidebar.button("Run Training"):
        run_pipeline()


def run_pipeline():
    config = load_config()
    df = load_dataset(config.DATASET_PATH)
    X_df, y = get_feature_target(df, config.SELECTED_FEATURES, config.TARGET_COLUMN)
    X_df = fill_numeric_missing(X_df)
    X_df = fill_categorical_missing(X_df, config.CATEGORICAL_FEATURES)
    X_df = encode_categoricals(X_df, config.CATEGORICAL_FEATURES)
    class_1 = config.TARGET_CLASSES[config.SELECTED_CLASSES[0]]
    class_2 = config.TARGET_CLASSES[config.SELECTED_CLASSES[1]]
    model_class = config.MODELS[config.SELECTED_MODEL]

    mask = (y == class_1) | (y == class_2)
    X_pair = X_df[mask].copy()
    y_pair = y[mask].copy()

    X_shuffled, y_shuffled = shuffle(X_pair, y_pair, random_state=42)
    y_shuffled = np.where(y_shuffled == class_1, 1, -1)

    X_train, X_test, y_train, y_test = train_test_split(
        X_shuffled.values,
        y_shuffled,
        test_size=config.TEST_SIZE,
        random_state=42,
        stratify=y_shuffled
    )

    X_train, X_test, X_mean, X_std = standardize_train_test(X_train, X_test)

    model = model_class(
        learning_rate=config.LEARNING_RATE,
        n_iters=config.N_ITERS,
        bias=config.BIAS,
        mse_threshold=config.MSE_THRESHOLD
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    confusion_matrix = generate_confusion_matrix(y_test, predictions)
    acc = accuracy_score(y_test, predictions)

    st.success(f"Model trained successfully using **{config.SELECTED_MODEL}**.")
    st.write(f"**Classes:** {class_1} vs {class_2}")
    st.write(f"**Features:** {config.SELECTED_FEATURES}")
    st.write(f"**Accuracy:** {acc * 100:.2f}%")
    st.write("**Confusion Matrix:**")
    st.write(confusion_matrix)


if __name__ == "__main__":
    main()
