from __future__ import annotations

import glob
from pathlib import Path

import cv2
import numpy as np
from tensorflow import keras


PROJECT_DIR = Path(__file__).resolve().parent
REAL_BLANK_DIR = PROJECT_DIR / "data" / "blank_cells"


def _load_single_source(source: str, exclude: bool):
    if source in {"mnist", "fonts"}:
        import prepare_data as base_data
        return base_data.get_data(source, exclude=exclude)

    if source == "hoda":
        try:
            import prepare_data_hoda as base_data
        except ImportError as exc:
            raise ImportError(
                "prepare_data_hoda.py is required for Hoda data."
            ) from exc
        return base_data.get_data("hoda", exclude=exclude)

    raise ValueError(f"Unsupported source: {source}")


def _load_base_data(data_choice: str, exclude: bool):
    choice = data_choice.lower()
    source_map = {
        "mnist": ["mnist"],
        "fonts": ["fonts"],
        "both": ["mnist", "fonts"],
        "hoda": ["hoda"],
        "all": ["mnist", "fonts", "hoda"],
    }

    if choice not in source_map:
        raise ValueError(
            "Invalid data choice. Use one of: "
            "'mnist', 'fonts', 'both', 'hoda', or 'all'."
        )

    loaded = [_load_single_source(source, exclude) for source in source_map[choice]]

    # Sanitize each source before concatenation. This also repairs the old
    # font loader's accidental (N, 28, 28, 1, 1) output.
    sanitized = []
    for dataset in loaded:
        x_train, x_val, x_test, y_train, y_val, y_test = dataset
        sanitized.append((
            _ensure_image_shape_and_scale(x_train),
            _ensure_image_shape_and_scale(x_val),
            _ensure_image_shape_and_scale(x_test),
            np.asarray(y_train),
            np.asarray(y_val),
            np.asarray(y_test),
        ))

    if len(sanitized) == 1:
        return sanitized[0]

    return tuple(
        np.concatenate([dataset[index] for dataset in sanitized], axis=0)
        for index in range(6)
    )

def _ensure_image_shape_and_scale(images: np.ndarray) -> np.ndarray:
    images = np.asarray(images)

    while images.ndim > 4 and images.shape[-1] == 1:
        images = np.squeeze(images, axis=-1)

    if images.ndim == 3:
        images = np.expand_dims(images, axis=-1)

    if images.ndim != 4 or images.shape[1:] != (28, 28, 1):
        raise ValueError(
            f"Expected image shape (N, 28, 28, 1), received {images.shape}."
        )

    images = images.astype("float32")
    if images.size and float(images.max()) > 1.0:
        images /= 255.0

    return np.clip(images, 0.0, 1.0)


def _convert_digit_labels_to_10_classes(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)

    if labels.ndim != 2:
        raise ValueError(f"Expected 2D one-hot labels, received {labels.shape}.")

    if labels.shape[1] == 9:
        digit_numbers = np.argmax(labels, axis=1) + 1
        return keras.utils.to_categorical(
            digit_numbers,
            num_classes=10,
        ).astype("float32")

    if labels.shape[1] == 10:
        return labels.astype("float32")

    raise ValueError(
        f"Expected 9- or 10-class labels, received {labels.shape[1]} classes."
    )


def _generate_synthetic_blank_images(count: int, seed: int) -> np.ndarray:
    if count <= 0:
        return np.empty((0, 28, 28, 1), dtype="float32")

    rng = np.random.default_rng(seed)
    images = np.zeros((count, 28, 28), dtype="float32")

    for index in range(count):
        image = rng.normal(loc=0.0, scale=0.012, size=(28, 28))
        image = np.clip(image, 0.0, 0.05)

        number_of_lines = int(rng.integers(0, 4))
        for _ in range(number_of_lines):
            intensity = float(rng.uniform(0.08, 0.40))
            thickness = int(rng.integers(1, 3))

            if rng.random() < 0.5:
                row = int(rng.choice([0, 1, 2, 25, 26, 27]))
                start = max(0, row - thickness + 1)
                end = min(28, row + thickness)
                image[start:end, :] = np.maximum(
                    image[start:end, :],
                    intensity,
                )
            else:
                column = int(rng.choice([0, 1, 2, 25, 26, 27]))
                start = max(0, column - thickness + 1)
                end = min(28, column + thickness)
                image[:, start:end] = np.maximum(
                    image[:, start:end],
                    intensity,
                )

        for _ in range(int(rng.integers(0, 5))):
            y = int(rng.integers(3, 25))
            x = int(rng.integers(3, 25))
            image[y, x] = float(rng.uniform(0.05, 0.25))

        images[index] = np.clip(image, 0.0, 1.0)

    return np.expand_dims(images, axis=-1)


def _load_real_blank_images() -> np.ndarray:
    patterns = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    paths: list[str] = []

    for pattern in patterns:
        paths.extend(glob.glob(str(REAL_BLANK_DIR / pattern)))

    loaded = []
    for path in sorted(paths):
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue

        image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)
        image = image.astype("float32") / 255.0

        if float(image.mean()) > 0.5:
            image = 1.0 - image

        loaded.append(np.expand_dims(np.clip(image, 0.0, 1.0), axis=-1))

    if not loaded:
        return np.empty((0, 28, 28, 1), dtype="float32")

    return np.asarray(loaded, dtype="float32")


def _blank_target_count(y_10: np.ndarray, ratio: float) -> int:
    labels = np.argmax(y_10, axis=1)
    digit_counts = [int(np.sum(labels == digit)) for digit in range(1, 10)]
    positive_counts = [count for count in digit_counts if count > 0]

    if not positive_counts:
        raise ValueError("No digit samples were found.")

    return max(1, int(round(float(np.median(positive_counts)) * ratio)))


def _split_real_blanks(
    real_blanks: np.ndarray,
    target_counts: tuple[int, int, int],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    if len(real_blanks):
        indices = rng.permutation(len(real_blanks))
        real_blanks = real_blanks[indices]

    train_target, val_target, test_target = target_counts
    total_target = train_target + val_target + test_target
    usable = real_blanks[:total_target]

    train_end = min(train_target, len(usable))
    val_end = min(train_target + val_target, len(usable))

    return usable[:train_end], usable[train_end:val_end], usable[val_end:]


def _make_blank_split(
    real_images: np.ndarray,
    target_count: int,
    seed: int,
) -> np.ndarray:
    missing = max(0, target_count - len(real_images))
    synthetic = _generate_synthetic_blank_images(missing, seed=seed)

    if len(real_images) == 0:
        return synthetic
    if len(synthetic) == 0:
        return real_images[:target_count]

    return np.concatenate((real_images, synthetic), axis=0)


def _append_blank_class(
    x: np.ndarray,
    y: np.ndarray,
    blanks: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    blank_labels = keras.utils.to_categorical(
        np.zeros(len(blanks), dtype=np.int64),
        num_classes=10,
    ).astype("float32")

    combined_x = np.concatenate((x, blanks), axis=0)
    combined_y = np.concatenate((y, blank_labels), axis=0)

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(combined_x))
    return combined_x[order], combined_y[order]


def get_data(
    data_choice: str,
    exclude: bool = True,
    blank_ratio: float = 1.0,
    random_state: int = 2026,
):
    if blank_ratio <= 0:
        raise ValueError("blank_ratio must be greater than zero.")

    x_train, x_val, x_test, y_train, y_val, y_test = _load_base_data(
        data_choice=data_choice,
        exclude=exclude,
    )

    x_train = _ensure_image_shape_and_scale(x_train)
    x_val = _ensure_image_shape_and_scale(x_val)
    x_test = _ensure_image_shape_and_scale(x_test)

    y_train = _convert_digit_labels_to_10_classes(y_train)
    y_val = _convert_digit_labels_to_10_classes(y_val)
    y_test = _convert_digit_labels_to_10_classes(y_test)

    targets = (
        _blank_target_count(y_train, blank_ratio),
        _blank_target_count(y_val, blank_ratio),
        _blank_target_count(y_test, blank_ratio),
    )

    real_blanks = _load_real_blank_images()
    real_train, real_val, real_test = _split_real_blanks(
        real_blanks,
        target_counts=targets,
        seed=random_state,
    )

    blank_train = _make_blank_split(real_train, targets[0], random_state + 1)
    blank_val = _make_blank_split(real_val, targets[1], random_state + 2)
    blank_test = _make_blank_split(real_test, targets[2], random_state + 3)

    x_train, y_train = _append_blank_class(
        x_train, y_train, blank_train, random_state + 11
    )
    x_val, y_val = _append_blank_class(
        x_val, y_val, blank_val, random_state + 12
    )
    x_test, y_test = _append_blank_class(
        x_test, y_test, blank_test, random_state + 13
    )

    return x_train, x_val, x_test, y_train, y_val, y_test
