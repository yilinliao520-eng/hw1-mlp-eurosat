import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import accuracy_score, confusion_matrix


CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]


@dataclass
class Config:
    data_dir: str = "EuroSAT_RGB"
    out_dir: str = "outputs"
    image_size: int = 32
    batch_size: int = 128
    epochs: int = 18
    lr: float = 0.08
    lr_decay: float = 0.92
    weight_decay: float = 1e-4
    hidden_dims: tuple = (128, 64)
    activation: str = "relu"
    seed: int = 42


def one_hot(y, num_classes):
    out = np.zeros((len(y), num_classes), dtype=np.float32)
    out[np.arange(len(y)), y] = 1.0
    return out


def load_dataset(data_dir, image_size):
    xs, ys, paths = [], [], []
    for label, cls in enumerate(CLASSES):
        cls_dir = Path(data_dir) / cls
        for path in sorted(cls_dir.glob("*.jpg")):
            img = Image.open(path).convert("RGB").resize((image_size, image_size), Image.BILINEAR)
            xs.append(np.asarray(img, dtype=np.float32).reshape(-1) / 255.0)
            ys.append(label)
            paths.append(str(path))
    return np.stack(xs), np.asarray(ys, dtype=np.int64), np.asarray(paths)


def stratified_split(y, train_ratio=0.7, val_ratio=0.15, seed=42):
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_train = int(len(idx) * train_ratio)
        n_val = int(len(idx) * val_ratio)
        train_idx.extend(idx[:n_train])
        val_idx.extend(idx[n_train:n_train + n_val])
        test_idx.extend(idx[n_train + n_val:])
    return [rng.permutation(np.asarray(v, dtype=np.int64)) for v in (train_idx, val_idx, test_idx)]


class MLP:
    def __init__(self, input_dim, hidden_dims, num_classes, activation="relu", seed=0):
        self.activation_name = activation
        self.rng = np.random.default_rng(seed)
        dims = [input_dim] + list(hidden_dims) + [num_classes]
        self.params = {}
        for i in range(len(dims) - 1):
            scale = np.sqrt(2.0 / dims[i]) if activation == "relu" else np.sqrt(1.0 / dims[i])
            self.params[f"W{i+1}"] = (self.rng.normal(0, scale, size=(dims[i], dims[i+1]))).astype(np.float32)
            self.params[f"b{i+1}"] = np.zeros((1, dims[i+1]), dtype=np.float32)

    def _activation(self, z):
        if self.activation_name == "relu":
            return np.maximum(z, 0)
        if self.activation_name == "tanh":
            return np.tanh(z)
        if self.activation_name == "sigmoid":
            return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        raise ValueError(f"unknown activation {self.activation_name}")

    def _activation_grad(self, a):
        if self.activation_name == "relu":
            return (a > 0).astype(np.float32)
        if self.activation_name == "tanh":
            return 1.0 - a * a
        if self.activation_name == "sigmoid":
            return a * (1.0 - a)
        raise ValueError(f"unknown activation {self.activation_name}")

    def forward(self, x):
        caches = {"A0": x}
        a = x
        num_layers = len(self.params) // 2
        for i in range(1, num_layers):
            z = a @ self.params[f"W{i}"] + self.params[f"b{i}"]
            a = self._activation(z)
            caches[f"Z{i}"] = z
            caches[f"A{i}"] = a
        logits = a @ self.params[f"W{num_layers}"] + self.params[f"b{num_layers}"]
        caches[f"Z{num_layers}"] = logits
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        return probs, caches

    def loss_and_grads(self, x, y, weight_decay):
        probs, caches = self.forward(x)
        n = len(y)
        y_onehot = one_hot(y, probs.shape[1])
        loss = -np.log(probs[np.arange(n), y] + 1e-12).mean()
        for key, value in self.params.items():
            if key.startswith("W"):
                loss += 0.5 * weight_decay * np.sum(value * value)

        grads = {}
        num_layers = len(self.params) // 2
        dz = (probs - y_onehot) / n
        for i in range(num_layers, 0, -1):
            a_prev = caches[f"A{i-1}"]
            grads[f"W{i}"] = a_prev.T @ dz + weight_decay * self.params[f"W{i}"]
            grads[f"b{i}"] = dz.sum(axis=0, keepdims=True)
            if i > 1:
                da_prev = dz @ self.params[f"W{i}"].T
                dz = da_prev * self._activation_grad(caches[f"A{i-1}"])
        return loss, grads

    def predict(self, x, batch_size=1024):
        preds = []
        for start in range(0, len(x), batch_size):
            probs, _ = self.forward(x[start:start + batch_size])
            preds.append(np.argmax(probs, axis=1))
        return np.concatenate(preds)

    def save(self, path, meta):
        np.savez_compressed(path, **self.params, meta=json.dumps(meta, ensure_ascii=False))

    @classmethod
    def load(cls, path):
        data = np.load(path, allow_pickle=True)
        meta = json.loads(str(data["meta"]))
        model = cls(meta["input_dim"], tuple(meta["hidden_dims"]), meta["num_classes"], meta["activation"], meta["seed"])
        for key in model.params:
            model.params[key] = data[key]
        return model, meta


def evaluate(model, x, y):
    pred = model.predict(x)
    return accuracy_score(y, pred), pred


def train_one(config, x_train, y_train, x_val, y_val):
    model = MLP(x_train.shape[1], config.hidden_dims, len(CLASSES), config.activation, config.seed)
    rng = np.random.default_rng(config.seed)
    best_params = {k: v.copy() for k, v in model.params.items()}
    best_val = -1.0
    history = {"train_loss": [], "train_acc": [], "val_acc": [], "lr": []}
    lr = config.lr
    for epoch in range(config.epochs):
        order = rng.permutation(len(x_train))
        total_loss = 0.0
        for start in range(0, len(x_train), config.batch_size):
            idx = order[start:start + config.batch_size]
            loss, grads = model.loss_and_grads(x_train[idx], y_train[idx], config.weight_decay)
            for key in model.params:
                model.params[key] -= lr * grads[key].astype(np.float32)
            total_loss += loss * len(idx)
        train_acc, _ = evaluate(model, x_train, y_train)
        val_acc, _ = evaluate(model, x_val, y_val)
        history["train_loss"].append(total_loss / len(x_train))
        history["train_acc"].append(float(train_acc))
        history["val_acc"].append(float(val_acc))
        history["lr"].append(float(lr))
        if val_acc > best_val:
            best_val = float(val_acc)
            best_params = {k: v.copy() for k, v in model.params.items()}
        print(
            f"epoch {epoch+1:02d}/{config.epochs} "
            f"loss={history['train_loss'][-1]:.4f} "
            f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} lr={lr:.5f}",
            flush=True,
        )
        lr *= config.lr_decay
    model.params = best_params
    return model, history, best_val


def plot_curves(history, out_dir):
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(epochs, history["train_loss"], marker="o")
    ax[0].set_title("Training Loss")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Cross-Entropy + L2")
    ax[0].grid(True, alpha=0.3)
    ax[1].plot(epochs, history["train_acc"], marker="o", label="Train")
    ax[1].plot(epochs, history["val_acc"], marker="o", label="Validation")
    ax[1].set_title("Accuracy")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)
    fig.tight_layout()
    path = Path(out_dir) / "training_curves.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_confusion(cm, out_dir):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES)), labels=CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES)), labels=CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix on Test Set")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = Path(out_dir) / "confusion_matrix.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_weights(model, config, out_dir):
    w = model.params["W1"]
    cols = 8
    rows = 4
    n = min(cols * rows, w.shape[1])
    fig, axes = plt.subplots(rows, cols, figsize=(10, 5))
    for i, ax in enumerate(axes.ravel()):
        ax.axis("off")
        if i < n:
            img = w[:, i].reshape(config.image_size, config.image_size, 3)
            lo, hi = np.percentile(img, [2, 98])
            img = np.clip((img - lo) / (hi - lo + 1e-8), 0, 1)
            ax.imshow(img)
            ax.set_title(f"h{i}", fontsize=7)
    fig.suptitle("First Hidden Layer Weight Patterns", fontsize=12)
    fig.tight_layout()
    path = Path(out_dir) / "first_layer_weights.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_errors(x, y, pred, paths, config, out_dir, max_items=12):
    wrong = np.where(y != pred)[0][:max_items]
    if len(wrong) == 0:
        wrong = np.arange(min(max_items, len(y)))
    cols = 4
    rows = int(np.ceil(len(wrong) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(11, 3 * rows))
    axes = np.asarray(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for ax, idx in zip(axes, wrong):
        img = x[idx].reshape(config.image_size, config.image_size, 3)
        ax.imshow(np.clip(img, 0, 1))
        ax.set_title(f"T: {CLASSES[y[idx]]}\nP: {CLASSES[pred[idx]]}", fontsize=8)
    fig.suptitle("Representative Error Cases", fontsize=12)
    fig.tight_layout()
    path = Path(out_dir) / "error_cases.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def make_report(config, search_results, best_result, test_acc, cm, figure_paths, repo_url, drive_url):
    out_dir = Path(config.out_dir)
    report_path = out_dir / "HW1_Report_25210980075_廖燚吝.pdf"
    with PdfPages(report_path) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.08, 0.94, "HW1: NumPy MLP for EuroSAT Land-Cover Classification", fontsize=17, weight="bold")
        fig.text(0.08, 0.90, "Student: Liao Yilin    Student ID: 25210980075", fontsize=12)
        body = (
            "Task. This work builds a three-layer MLP classifier from scratch for EuroSAT RGB image classification. "
            "The implementation uses NumPy only for matrix computation and does not use PyTorch, TensorFlow, JAX, "
            "or automatic differentiation.\n\n"
            f"Data. Images are resized to {config.image_size}x{config.image_size}, flattened, normalized with the "
            "training-set mean and standard deviation, and split stratified into 70% train, 15% validation, "
            "and 15% test subsets.\n\n"
            f"Model. The selected model uses hidden dimensions {best_result['hidden_dims']}, "
            f"{best_result['activation']} activation, mini-batch SGD, cross-entropy loss, L2 weight decay, "
            "learning-rate decay, and validation-based best checkpoint selection.\n\n"
            f"Final test accuracy: {test_acc:.4f}\n"
            f"GitHub repo: {repo_url}\n"
            f"Model weights: {drive_url}\n"
        )
        fig.text(0.08, 0.84, body, fontsize=10, va="top", wrap=True)
        fig.text(0.08, 0.43, "Hyperparameter search summary", fontsize=13, weight="bold")
        rows = ["activation | hidden_dims | lr | weight_decay | best_val_acc"]
        for r in search_results:
            rows.append(f"{r['activation']} | {r['hidden_dims']} | {r['lr']} | {r['weight_decay']} | {r['best_val_acc']:.4f}")
        fig.text(0.08, 0.40, "\n".join(rows), fontsize=9, family="monospace", va="top")
        fig.text(0.08, 0.20, "Observation. ReLU converged fastest in this experiment. Confusions mostly occur among visually similar land-cover types such as PermanentCrop, AnnualCrop, Pasture, Highway, and River.", fontsize=10, wrap=True)
        ax0 = fig.add_axes([0, 0, 1, 1])
        ax0.axis("off")
        pdf.savefig(fig)
        plt.close(fig)
        for title, path in figure_paths:
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.text(0.08, 0.95, title, fontsize=14, weight="bold")
            img = plt.imread(path)
            ax = fig.add_axes([0.06, 0.08, 0.88, 0.82])
            ax.imshow(img)
            ax.axis("off")
            pdf.savefig(fig)
            plt.close(fig)
    return report_path


def run(args):
    config = Config(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        image_size=args.image_size,
        epochs=args.epochs,
        seed=args.seed,
    )
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...", flush=True)
    x, y, paths = load_dataset(config.data_dir, config.image_size)
    train_idx, val_idx, test_idx = stratified_split(y, seed=config.seed)
    x_train_raw, y_train = x[train_idx], y[train_idx]
    x_val_raw, y_val = x[val_idx], y[val_idx]
    x_test_raw, y_test = x[test_idx], y[test_idx]
    test_paths = paths[test_idx]
    mean = x_train_raw.mean(axis=0, keepdims=True)
    std = x_train_raw.std(axis=0, keepdims=True) + 1e-6
    x_train = ((x_train_raw - mean) / std).astype(np.float32)
    x_val = ((x_val_raw - mean) / std).astype(np.float32)
    x_test = ((x_test_raw - mean) / std).astype(np.float32)

    search_space = [
        {"activation": "relu", "hidden_dims": (128, 64), "lr": 0.08, "weight_decay": 1e-4},
        {"activation": "relu", "hidden_dims": (256, 128), "lr": 0.05, "weight_decay": 5e-4},
        {"activation": "tanh", "hidden_dims": (128, 64), "lr": 0.04, "weight_decay": 1e-4},
        {"activation": "sigmoid", "hidden_dims": (128, 64), "lr": 0.08, "weight_decay": 1e-4},
    ]
    if args.fast:
        search_space = search_space[:2]
        config.epochs = min(config.epochs, 8)

    search_results = []
    best = None
    best_model = None
    best_history = None
    for i, hp in enumerate(search_space, 1):
        cfg = Config(**{**config.__dict__, **hp, "seed": config.seed + i})
        print(f"\n=== Run {i}/{len(search_space)}: {hp} ===", flush=True)
        model, history, best_val = train_one(cfg, x_train, y_train, x_val, y_val)
        result = {**hp, "best_val_acc": best_val}
        search_results.append(result)
        if best is None or best_val > best["best_val_acc"]:
            best = result
            best_model = model
            best_history = history
            config = cfg

    test_acc, test_pred = evaluate(best_model, x_test, y_test)
    cm = confusion_matrix(y_test, test_pred, labels=np.arange(len(CLASSES)))
    meta = {
        "student": "廖燚吝",
        "student_id": "25210980075",
        "github": "yilinliao520-eng",
        "google_account": "yilinliao520@gmail.com",
        "input_dim": x_train.shape[1],
        "num_classes": len(CLASSES),
        "classes": CLASSES,
        "hidden_dims": list(best["hidden_dims"]),
        "activation": best["activation"],
        "seed": config.seed,
        "image_size": config.image_size,
        "test_accuracy": float(test_acc),
    }
    weights_path = out_dir / "best_mlp_weights_25210980075.npz"
    best_model.save(weights_path, meta)
    (out_dir / "metrics.json").write_text(json.dumps({
        "search_results": search_results,
        "best": best,
        "test_accuracy": float(test_acc),
        "confusion_matrix": cm.tolist(),
    }, ensure_ascii=False, indent=2))

    curve_path = plot_curves(best_history, out_dir)
    cm_path = plot_confusion(cm, out_dir)
    weights_vis_path = plot_weights(best_model, config, out_dir)
    err_path = plot_errors(x_test_raw, y_test, test_pred, test_paths, config, out_dir)
    repo_url = "https://github.com/yilinliao520-eng/hw1-mlp-eurosat"
    drive_url = "https://drive.google.com/drive/folders/TO_BE_UPLOADED_BY_yilinliao520"
    report_path = make_report(
        config,
        search_results,
        best,
        test_acc,
        cm,
        [
            ("Training and Validation Curves", curve_path),
            ("Confusion Matrix", cm_path),
            ("First Hidden Layer Weight Visualization", weights_vis_path),
            ("Error Analysis Examples", err_path),
        ],
        repo_url,
        drive_url,
    )
    print("\nDone.")
    print(f"Best validation accuracy: {best['best_val_acc']:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Weights: {weights_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="EuroSAT_RGB")
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fast", action="store_true")
    run(parser.parse_args())
