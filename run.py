import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from preprocessing.transforms import encode_categoricals, fill_categorical_missing, fill_numeric_missing, standardize_train_test
from utils.utils import generate_confusion_matrix, load_config
from data.loader import load_dataset, get_feature_target
from sklearn.utils import shuffle
# TODO: clean the enums in config file
def main():
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
            X_shuffled.values, y_shuffled, 
            test_size=config.TEST_SIZE, 
            random_state=42,
            stratify=y_shuffled
        )

    X_train, X_test, X_mean, X_std = standardize_train_test(X_train, X_test)

    model = model_class(
            learning_rate=config.LEARNING_RATE,
            n_iters=config.N_ITERS,
            bias=config.BIAS,
            mse_threshold= config.MSE_THRESHOLD
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    # TODO: Scatter plot on test set with predictions vs true labels

    confusion_matrix = generate_confusion_matrix(y_test, predictions)

    print("Confusion Matrix:", confusion_matrix)
    acc = accuracy_score(y_test, predictions)

    print(f"{class_1} vs {class_2}, Accuracy: {acc * 100:.2f}%")

if __name__ == "__main__":
    main()

