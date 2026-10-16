"""Train the retinal disease classifier used by app.py.

Produces ``retinal_disease_model.h5``: a ResNet50 (ImageNet pre-trained) transfer-learning
model that takes 224x224 RGB images prepared with ``resnet50.preprocess_input`` and
returns 8 softmax scores in the same order as ``DISEASE_LABELS`` in app.py.

Expected dataset layout (one sub-folder per class; folder names must match exactly):

    data/
    ├── train/
    │   ├── Normal/
    │   ├── Diabetic Retinopathy/
    │   ├── Glaucoma/
    │   ├── Cataract/
    │   ├── AMD/
    │   ├── Hypertension/
    │   ├── Myopia/
    │   └── Other/
    └── test/            # optional, same structure; used for the final evaluation

If ``data/test`` does not exist, a part of ``data/train`` is held out for validation and
the final report is computed on that validation split instead.

Usage:
    python train.py --data-dir data
    python train.py --data-dir data --epochs 15 --fine-tune-epochs 10 --batch-size 16

Educational project only - the resulting model is not a medical device.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

# IMPORTANT: must be identical (names and order) to DISEASE_LABELS in app.py.
DISEASE_LABELS = [
    "Normal",
    "Diabetic Retinopathy",
    "Glaucoma",
    "Cataract",
    "AMD",
    "Hypertension",
    "Myopia",
    "Other",
]
IMAGE_SIZE = (224, 224)
SEED = 42


def parse_args():
    p = argparse.ArgumentParser(description="Train the retinal disease classifier.")
    p.add_argument("--data-dir", type=Path, default=Path("data"), help="Folder with train/ (and optional test/).")
    p.add_argument("--output", type=Path, default=Path("retinal_disease_model.h5"), help="Where to save the model.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=15, help="Epochs with the ResNet50 base frozen.")
    p.add_argument("--fine-tune-epochs", type=int, default=10, help="Extra epochs with the top of the base unfrozen. 0 to skip.")
    p.add_argument("--fine-tune-layers", type=int, default=30, help="Number of final base layers to unfreeze.")
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--fine-tune-lr", type=float, default=1e-5)
    p.add_argument("--val-split", type=float, default=0.2, help="Validation fraction taken from train/.")
    return p.parse_args()


def check_class_folders(split_dir: Path):
    """Fail early if folder names don't match the labels the app expects."""
    found = {d.name for d in split_dir.iterdir() if d.is_dir()}
    missing = [c for c in DISEASE_LABELS if c not in found]
    extra = sorted(found - set(DISEASE_LABELS))
    if missing or extra:
        raise SystemExit(
            f"Class folders in '{split_dir}' do not match the expected labels.\n"
            f"  Missing: {missing}\n  Unexpected: {extra}\n"
            f"  Expected exactly: {DISEASE_LABELS}"
        )


def load_dataset(directory: Path, batch_size: int, **kwargs):
    # class_names fixes the label order to DISEASE_LABELS. Without it Keras sorts folders
    # alphabetically, which would NOT match the order used in app.py.
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="categorical",
        class_names=DISEASE_LABELS,
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        color_mode="rgb",
        **kwargs,
    )


def build_datasets(args):
    train_dir = args.data_dir / "train"
    test_dir = args.data_dir / "test"
    if not train_dir.is_dir():
        raise SystemExit(f"Training folder not found: {train_dir}")
    check_class_folders(train_dir)

    train_ds = load_dataset(
        train_dir, args.batch_size, validation_split=args.val_split, subset="training", seed=SEED
    )
    val_ds = load_dataset(
        train_dir, args.batch_size, validation_split=args.val_split, subset="validation", seed=SEED,
        shuffle=False,
    )

    test_ds = None
    if test_dir.is_dir():
        check_class_folders(test_dir)
        test_ds = load_dataset(test_dir, args.batch_size, shuffle=False)

    # Class weights from the training folders to counter class imbalance.
    counts = np.array(
        [len([f for f in (train_dir / c).rglob("*") if f.is_file()]) for c in DISEASE_LABELS], dtype=np.float64
    )
    if np.any(counts == 0):
        raise SystemExit(f"Some classes have no images: {dict(zip(DISEASE_LABELS, counts.astype(int)))}")
    weights = counts.sum() / (len(counts) * counts)
    class_weight = {i: float(w) for i, w in enumerate(weights)}
    print("Images per class:", dict(zip(DISEASE_LABELS, counts.astype(int))))

    augment = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ],
        name="augmentation",
    )
    autotune = tf.data.AUTOTUNE

    # Augmentation works on raw 0-255 pixels; ResNet50 preprocess_input is applied afterwards,
    # exactly as app.py does at inference time.
    train_ds = train_ds.map(
        lambda x, y: (preprocess_input(augment(x, training=True)), y), num_parallel_calls=autotune
    ).prefetch(autotune)
    val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=autotune).prefetch(autotune)
    if test_ds is not None:
        test_ds = test_ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=autotune).prefetch(autotune)

    return train_ds, val_ds, test_ds, class_weight


def build_model():
    base = ResNet50(weights="imagenet", include_top=False, input_shape=IMAGE_SIZE + (3,))
    base.trainable = False

    inputs = layers.Input(shape=IMAGE_SIZE + (3,))
    # training=False keeps BatchNorm layers in inference mode, also during fine-tuning.
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(len(DISEASE_LABELS), activation="softmax")(x)  # app.py expects probabilities
    return models.Model(inputs, outputs, name="retinal_resnet50"), base


def compile_model(model, lr):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )


def make_callbacks(output: Path):
    return [
        tf.keras.callbacks.ModelCheckpoint(output, monitor="val_loss", save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7, verbose=1),
    ]


def evaluate(model, dataset, name: str, output_dir: Path):
    """Print per-class precision/recall/F1 and a confusion matrix; save them next to the model."""
    y_true = np.concatenate([np.argmax(y.numpy(), axis=1) for _, y in dataset])
    y_pred = np.argmax(model.predict(dataset, verbose=0), axis=1)

    metrics = {"split": name, "accuracy": float(np.mean(y_true == y_pred))}
    print(f"\n=== Evaluation on {name} set ===\naccuracy: {metrics['accuracy']:.4f}")

    try:
        from sklearn.metrics import classification_report, confusion_matrix

        labels = list(range(len(DISEASE_LABELS)))
        print(classification_report(y_true, y_pred, labels=labels, target_names=DISEASE_LABELS, zero_division=0))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        print("Confusion matrix (rows = true, columns = predicted):\n", cm)
        metrics["classification_report"] = classification_report(
            y_true, y_pred, labels=labels, target_names=DISEASE_LABELS, zero_division=0, output_dict=True
        )
        metrics["confusion_matrix"] = cm.tolist()
    except ImportError:
        print("Install scikit-learn for per-class precision/recall/F1 and the confusion matrix.")

    with open(output_dir / "evaluation.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def main():
    args = parse_args()
    tf.keras.utils.set_random_seed(SEED)

    train_ds, val_ds, test_ds, class_weight = build_datasets(args)
    model, base = build_model()
    output_dir = args.output.resolve().parent
    callbacks = make_callbacks(args.output)

    # Stage 1: train the new classification head on top of the frozen ResNet50 base.
    compile_model(model, args.learning_rate)
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, class_weight=class_weight, callbacks=callbacks)

    # Stage 2: fine-tune the last few ResNet50 layers at a low learning rate.
    if args.fine_tune_epochs > 0:
        base.trainable = True
        for layer in base.layers[: -args.fine_tune_layers]:
            layer.trainable = False
        for layer in base.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False  # keep BN statistics stable with small batches
        compile_model(model, args.fine_tune_lr)
        model.fit(
            train_ds, validation_data=val_ds, epochs=args.fine_tune_epochs,
            class_weight=class_weight, callbacks=make_callbacks(args.output),
        )

    # Reload the best checkpoint (lowest val_loss) so what we evaluate is what we ship.
    best = tf.keras.models.load_model(args.output)
    with open(output_dir / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(DISEASE_LABELS, f, indent=2)

    if test_ds is not None:
        evaluate(best, test_ds, "test", output_dir)
    else:
        print("\nNo data/test folder found - reporting on the validation split instead.")
        evaluate(best, val_ds, "validation", output_dir)

    print(f"\nModel saved to {args.output}  (class order: {DISEASE_LABELS})")
    print("Copy it next to app.py and run:  streamlit run app.py")


if __name__ == "__main__":
    main()
