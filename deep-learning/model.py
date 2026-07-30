"""
🧠 Deep Learning — Model Architecture
========================================
CNN model definition for MNIST handwritten digit recognition.
Reusable across train.py, predict.py, and evaluate.py.
"""

from tensorflow.keras import layers, models


def build_mnist_cnn(input_shape: tuple = (28, 28, 1), num_classes: int = 10):
    """
    Build a Convolutional Neural Network for MNIST digit classification.

    Architecture:
        Conv2D(32, 3x3) → ReLU → Conv2D(64, 3x3) → ReLU → MaxPool(2x2)
        → Dropout(0.25) → Flatten → Dense(128) → ReLU → Dropout(0.5)
        → Dense(10, softmax)

    Args:
        input_shape: Shape of input images (height, width, channels).
        num_classes: Number of output classes.

    Returns:
        Compiled Keras model.
    """
    model = models.Sequential([
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape,
                      name="conv1"),
        layers.Conv2D(64, (3, 3), activation="relu", name="conv2"),
        layers.MaxPooling2D(pool_size=(2, 2), name="pool1"),
        layers.Dropout(0.25, name="dropout1"),

        # Flatten and dense layers
        layers.Flatten(name="flatten"),
        layers.Dense(128, activation="relu", name="dense1"),
        layers.Dropout(0.5, name="dropout2"),

        # Output layer
        layers.Dense(num_classes, activation="softmax", name="output"),
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def get_model_summary(model) -> str:
    """Return the model summary as a string."""
    summary_lines = []
    model.summary(print_fn=lambda x: summary_lines.append(x))
    return "\n".join(summary_lines)


if __name__ == "__main__":
    # Quick test: build and display model
    model = build_mnist_cnn()
    print("🧠 MNIST CNN Architecture\n")
    model.summary()
    total_params = model.count_params()
    print(f"\n📊 Total parameters: {total_params:,}")
