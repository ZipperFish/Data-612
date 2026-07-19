"""
Train the wine-food pairing recommender (SGD Matrix Factorization / "Funk SVD")
and save the model artifacts needed by the prediction API.

This reproduces the modeling approach from Project 3 (matrix factorization
methods) so the trained model can be served from an Azure-hosted API.

Usage:
    python train_model.py
Output:
    model_artifacts.joblib  -- everything the API needs to make predictions
"""
import numpy as np
import pandas as pd
import joblib

DATA_PATH = "wine_food_pairings.csv"
ARTIFACT_PATH = "model_artifacts.joblib"

SEED = 612
K = 8
LR = 0.02
REG = 0.1
N_EPOCHS = 200


class SGDMatrixFactorization:
    def __init__(self, n_rows, n_cols, k=10, lr=0.02, reg=0.05, n_epochs=200, seed=612):
        rng = np.random.default_rng(seed)
        self.k = k
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self.P = rng.normal(scale=0.1, size=(n_rows, k))
        self.Q = rng.normal(scale=0.1, size=(n_cols, k))
        self.b_row = np.zeros(n_rows)
        self.b_col = np.zeros(n_cols)
        self.mu = 0.0

    def fit(self, row_idx, col_idx, values, verbose=True):
        self.mu = values.mean()
        n = len(values)
        order = np.arange(n)
        history = []

        for epoch in range(self.n_epochs):
            rng = np.random.default_rng(1000 + epoch)
            rng.shuffle(order)
            sq_err_sum = 0.0

            for idx in order:
                r, c, val = row_idx[idx], col_idx[idx], values[idx]
                pred = self.mu + self.b_row[r] + self.b_col[c] + self.P[r] @ self.Q[c]
                err = val - pred
                sq_err_sum += err ** 2

                self.b_row[r] += self.lr * (err - self.reg * self.b_row[r])
                self.b_col[c] += self.lr * (err - self.reg * self.b_col[c])
                p_r = self.P[r].copy()
                self.P[r] += self.lr * (err * self.Q[c] - self.reg * self.P[r])
                self.Q[c] += self.lr * (err * p_r - self.reg * self.Q[c])

            train_rmse = np.sqrt(sq_err_sum / n)
            history.append(train_rmse)
            if verbose and (epoch % 25 == 0 or epoch == self.n_epochs - 1):
                print(f"epoch {epoch+1:>3}/{self.n_epochs}  train RMSE: {train_rmse:.4f}")
        return history

    def predict(self, row_idx, col_idx):
        preds = self.mu + self.b_row[row_idx] + self.b_col[col_idx] + np.sum(
            self.P[row_idx] * self.Q[col_idx], axis=1
        )
        return np.clip(preds, 1, 5)


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def mae(y_true, y_pred):
    return np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)))


def main():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)

    # Collapse repeated (food, wine) rows into one averaged pair per combo
    pair_table = (
        df.groupby(["food_item", "wine_type"])["pairing_quality"]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": "avg_quality", "std": "quality_std", "count": "n_scored"})
        .reset_index()
    )
    print(f"Aggregated pair table: {len(pair_table)} unique (food, wine) pairs")

    # Train/test split (matches notebook: 80/20, same seed)
    pair_table_shuffled = pair_table.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    n_test = int(len(pair_table_shuffled) * 0.2)
    test_pairs = pair_table_shuffled.iloc[:n_test].copy()
    train_pairs = pair_table_shuffled.iloc[n_test:].copy()
    print(f"Train: {len(train_pairs)} pairs   Test: {len(test_pairs)} pairs")

    # Index mappings
    food_items_sorted = np.sort(pair_table["food_item"].unique())
    wine_types_sorted = np.sort(pair_table["wine_type"].unique())
    food_idx = {f: i for i, f in enumerate(food_items_sorted)}
    wine_idx = {w: i for i, w in enumerate(wine_types_sorted)}

    train_f = train_pairs["food_item"].map(food_idx).values
    train_w = train_pairs["wine_type"].map(wine_idx).values
    train_y = train_pairs["avg_quality"].values

    test_f = test_pairs["food_item"].map(food_idx).values
    test_w = test_pairs["wine_type"].map(wine_idx).values
    test_y = test_pairs["avg_quality"].values

    print("Training SGD Matrix Factorization model...")
    sgd_mf = SGDMatrixFactorization(
        n_rows=len(food_items_sorted),
        n_cols=len(wine_types_sorted),
        k=K, lr=LR, reg=REG, n_epochs=N_EPOCHS, seed=SEED,
    )
    sgd_mf.fit(train_f, train_w, train_y)

    test_preds = sgd_mf.predict(test_f, test_w)
    test_rmse = rmse(test_y, test_preds)
    test_mae = mae(test_y, test_preds)
    print(f"Final test RMSE: {test_rmse:.4f}   MAE: {test_mae:.4f}")

    # What each food/wine has already been scored against (used to filter
    # recommendations down to *unscored* pairs, same as the notebook)
    scored_by_food = train_pairs.groupby("food_item")["wine_type"].apply(set).to_dict()
    scored_by_wine = train_pairs.groupby("wine_type")["food_item"].apply(set).to_dict()

    artifacts = {
        "P": sgd_mf.P,
        "Q": sgd_mf.Q,
        "b_row": sgd_mf.b_row,
        "b_col": sgd_mf.b_col,
        "mu": sgd_mf.mu,
        "k": K, "lr": LR, "reg": REG, "n_epochs": N_EPOCHS,
        "food_idx": food_idx,
        "wine_idx": wine_idx,
        "food_items_sorted": food_items_sorted.tolist(),
        "wine_types_sorted": wine_types_sorted.tolist(),
        "scored_by_food": scored_by_food,
        "scored_by_wine": scored_by_wine,
        "test_rmse": test_rmse,
        "test_mae": test_mae,
    }
    joblib.dump(artifacts, ARTIFACT_PATH)
    print(f"Saved model artifacts to {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
