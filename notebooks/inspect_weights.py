import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("../artifacts/best_model.keras")

for layer in model.layers:
    if "dense" in layer.name.lower():
        weights = layer.get_weights()
        if len(weights) == 0:
            continue
        W, b = weights[0], weights[1]
        print(f"\nLayer: {layer.name}")
        print(f"  W shape={W.shape} | mean={W.mean():.6f} std={W.std():.6f} | isnan={np.isnan(W).any()}")
        print(f"  b shape={b.shape} | mean={b.mean():.6f} std={b.std():.6f} | min={b.min():.4f} max={b.max():.4f}")
