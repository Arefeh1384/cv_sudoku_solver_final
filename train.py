from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

import prepare_data_10class as prep_data


def build_ten_class_model() -> keras.Model:
    augmentation = keras.Sequential(
        [
            layers.RandomRotation(0.03, fill_mode="constant", fill_value=0.0),
            layers.RandomTranslation(0.06, 0.06, fill_mode="constant", fill_value=0.0),
            layers.RandomZoom((-0.06, 0.06), (-0.06, 0.06), fill_mode="constant", fill_value=0.0),
        ],
        name="data_augmentation",
    )

    return keras.Sequential(
        [
            keras.Input(shape=(28, 28, 1), name="cell_image"),
            augmentation,
            layers.Conv2D(32, (3, 3), activation="relu", name="conv_1"),
            layers.MaxPooling2D((2, 2), name="pool_1"),
            layers.Conv2D(64, (3, 3), activation="relu", name="conv_2"),
            layers.MaxPooling2D((2, 2), name="pool_2"),
            layers.Flatten(name="flatten"),
            layers.Dropout(0.5, name="dropout"),
            layers.Dense(10, activation="softmax", name="class_output"),
        ],
        name="sudoku_10class_transfer_model",
    )


def weighted_layers(model: keras.Model):
    convs = [layer for layer in model.layers if isinstance(layer, layers.Conv2D)]
    dense = [layer for layer in model.layers if isinstance(layer, layers.Dense)]
    if len(convs) < 2 or not dense:
        raise ValueError("Source model does not match the expected CNN architecture.")
    return convs[:2], dense[-1]


def transfer_weights(old_model: keras.Model, new_model: keras.Model) -> None:
    old_convs, old_dense = weighted_layers(old_model)
    new_convs, new_dense = weighted_layers(new_model)

    for old_layer, new_layer in zip(old_convs, new_convs):
        if [w.shape for w in old_layer.get_weights()] != [w.shape for w in new_layer.get_weights()]:
            raise ValueError(f"Convolution shape mismatch: {old_layer.name} -> {new_layer.name}")
        new_layer.set_weights(old_layer.get_weights())

    old_kernel, old_bias = old_dense.get_weights()
    new_kernel, new_bias = new_dense.get_weights()

    if old_kernel.shape[1] != 9 or new_kernel.shape[1] != 10:
        raise ValueError(
            f"Expected old Dense=9 and new Dense=10, received {old_kernel.shape} and {new_kernel.shape}."
        )
    if old_kernel.shape[0] != new_kernel.shape[0]:
        raise ValueError("Dense feature dimensions do not match.")

    new_kernel[:, 0] = 0.0
    new_bias[0] = 0.0
    new_kernel[:, 1:] = old_kernel
    new_bias[1:] = old_bias
    new_dense.set_weights([new_kernel, new_bias])


def compile_model(model: keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )


def freeze_features(model: keras.Model, freeze: bool) -> None:
    for layer in model.layers:
        if isinstance(layer, (layers.Conv2D, layers.MaxPooling2D)):
            layer.trainable = not freeze


def main(args: dict) -> None:
    old_path = Path(args["old_model"])
    out_path = Path(args["output_model"])
    out_dir = Path(args["output_dir"])

    if not old_path.exists():
        raise FileNotFoundError(f"Working 9-class model not found: {old_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading working 9-class model...")
    old_model = keras.models.load_model(old_path, compile=False)

    x_train, x_val, x_test, y_train, y_val, y_test = prep_data.get_data(
        data_choice=args["data"],
        exclude=True,
        blank_ratio=args["blank_ratio"],
        random_state=2026,
    )

    print("Train:", x_train.shape, y_train.shape)
    print("Validation:", x_val.shape, y_val.shape)
    print("Test:", x_test.shape, y_test.shape)

    model = build_ten_class_model()
    model(np.zeros((1, 28, 28, 1), dtype="float32"))
    transfer_weights(old_model, model)

    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=str(out_path),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    )

    print("\nStage 1: train the new blank/output head...")
    freeze_features(model, True)
    compile_model(model, 1e-3)
    history_head = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args["head_epochs"],
        batch_size=args["batch_size"],
        callbacks=[checkpoint],
        shuffle=True,
        verbose=1,
    )

    print("\nStage 2: fine-tune the complete ten-class model...")
    freeze_features(model, False)
    compile_model(model, 1e-4)
    callbacks = [
        checkpoint,
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1
        ),
    ]
    history_fine = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args["fine_tune_epochs"],
        batch_size=args["batch_size"],
        callbacks=callbacks,
        shuffle=True,
        verbose=1,
    )

    best_model = keras.models.load_model(out_path, compile=True)
    test_loss, test_accuracy = best_model.evaluate(x_test, y_test, verbose=1)

    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "output_shape": list(best_model.output_shape),
        "source_model": str(old_path),
        "blank_ratio": float(args["blank_ratio"]),
        "head_epochs_run": len(history_head.history["loss"]),
        "fine_tune_epochs_run": len(history_fine.history["loss"]),
    }
    (out_dir / "transfer_training_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    print("\nTransfer training complete.")
    print(f"Test accuracy: {test_accuracy:.4%}")
    print(f"Model saved at: {out_path}")
    print(f"Output shape: {best_model.output_shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--old_model",
        default="models/model_15_epochs_font_mnist_fixed.keras",
    )
    parser.add_argument(
        "--output_model",
        default="models/model_phase2_10class_transfer.keras",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/phase2_10class_transfer",
    )
    parser.add_argument(
        "--data",
        default="both",
        choices=["mnist", "fonts", "both", "hoda", "all"],
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--head_epochs", type=int, default=3)
    parser.add_argument("--fine_tune_epochs", type=int, default=12)
    parser.add_argument("--blank_ratio", type=float, default=0.6)
    main(vars(parser.parse_args()))
