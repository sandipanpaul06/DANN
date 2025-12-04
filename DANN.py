import numpy as np
import argparse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, BatchNormalization, Activation,
    Dropout, Flatten, Dense, concatenate
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# ============================================================
# Argparse
# ============================================================
### >>> CHANGE: user CLI options
parser = argparse.ArgumentParser(description="Train DANN model")

parser.add_argument("--src_sweep", required=True, help="Path to source sweep .npy")
parser.add_argument("--src_neut", required=True, help="Path to source neutral .npy")
parser.add_argument("--tgt_sweep", required=True, help="Path to target sweep .npy")
parser.add_argument("--tgt_neut", required=True, help="Path to target neutral .npy")

parser.add_argument("--src_train", type=int, default=5000, help="Source train samples per class")
parser.add_argument("--src_val", type=int, default=1000, help="Source val samples per class")
parser.add_argument("--tgt_train", type=int, default=5000, help="Target train samples per class")
parser.add_argument("--tgt_val", type=int, default=1000, help="Target val samples per class")

parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
parser.add_argument("--batch", type=int, default=32, help="Batch size")

args = parser.parse_args()

print("\n=== Loaded Arguments ===")
print(args)

# ============================================================
# Gradient Reversal Layer
# ============================================================
@tf.custom_gradient
def gradient_reversal(x, alpha):
    alpha = tf.cast(alpha, x.dtype)
    def grad(dy):
        return -alpha * dy, None
    return x, grad

class GradientReversalLayer(keras.layers.Layer):
    def __init__(self, alpha=1.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = tf.cast(alpha, tf.float32)

    def call(self, x):
        return gradient_reversal(x, self.alpha)

    def set_alpha(self, alpha):
        self.alpha = tf.cast(alpha, tf.float32)

# ============================================================
# Feature extractor (YOUR CNN)
# ============================================================
### >>> CHANGE: replacing previous 3-branch CNN
def build_feature_extractor(inputs):
    m = Conv2D(32, (3,3), padding='same')(inputs)
    m = BatchNormalization()(m)
    m = Activation('relu')(m)
    m = MaxPooling2D((2,2), strides=2)(m)

    m = Conv2D(32, (3,3), padding='same')(m)
    m = BatchNormalization()(m)
    m = Activation('relu')(m)
    m = Flatten()(m)

    features = Dense(128, activation='relu')(m)
    return features

# ============================================================
# Full DANN model
# ============================================================
def build_dann(input_shape, alpha=1.0):
    inp = Input(input_shape)

    features = build_feature_extractor(inp)

    # Label predictor
    label_out = Dense(1, activation="sigmoid", name="label")(features)

    # Domain classifier
    grl_layer = GradientReversalLayer(alpha=alpha)
    grl = grl_layer(features)
    domain_out = Dense(2, activation="softmax", name="domain")(grl)

    model = Model(inputs=inp, outputs=[label_out, domain_out])
    return model, grl_layer

# ============================================================
# Load data
# ============================================================
sweep_src = np.load(args.src_sweep)
neutral_src = np.load(args.src_neut)
sweep_tgt = np.load(args.tgt_sweep)
neutral_tgt = np.load(args.tgt_neut)

print("\n=== Loaded Datasets ===")
print("Source sweep:", sweep_src.shape)
print("Source neutral:", neutral_src.shape)
print("Target sweep:", sweep_tgt.shape)
print("Target neutral:", neutral_tgt.shape)

# ============================================================
# Auto-detect image shape
# ============================================================
### >>> CHANGE: automatically infer shape from .npy
img_h, img_w = sweep_src.shape[1], sweep_src.shape[2]
input_shape = (img_h, img_w, 1)

# Add channel dimension
sweep_src = sweep_src.reshape(-1, img_h, img_w, 1)
neutral_src = neutral_src.reshape(-1, img_h, img_w, 1)
sweep_tgt = sweep_tgt.reshape(-1, img_h, img_w, 1)
neutral_tgt = neutral_tgt.reshape(-1, img_h, img_w, 1)

# ============================================================
# Extract counts
# ============================================================
Ns_tr = args.src_train
Ns_val = args.src_val
Nt_tr = args.tgt_train
Nt_val = args.tgt_val

# ============================================================
# Build train/val/test splits
# ============================================================
# Source splits
sweep_src_train = sweep_src[:Ns_tr]
sweep_src_val = sweep_src[Ns_tr:Ns_tr+Ns_val]

neutral_src_train = neutral_src[:Ns_tr]
neutral_src_val = neutral_src[Ns_tr:Ns_tr+Ns_val]

# Target splits
sweep_tgt_train = sweep_tgt[:Nt_tr]
sweep_tgt_val = sweep_tgt[Nt_tr:Nt_tr+Nt_val]
sweep_tgt_test = sweep_tgt[Nt_tr+Nt_val:]

neutral_tgt_train = neutral_tgt[:Nt_tr]
neutral_tgt_val = neutral_tgt[Nt_tr:Nt_tr+Nt_val]
neutral_tgt_test = neutral_tgt[Nt_tr+Nt_val:]

print("\n=== Using image size:", input_shape, "===")

# ============================================================
# Assemble datasets
# ============================================================
X_train_src = np.concatenate([sweep_src_train, neutral_src_train])
y_train_src = np.array([1]*Ns_tr + [0]*Ns_tr)
d_train_src = np.zeros(len(X_train_src), dtype="int")

X_train_tgt = np.concatenate([sweep_tgt_train, neutral_tgt_train])
y_train_tgt = np.array([-1]*len(X_train_tgt))
d_train_tgt = np.ones(len(X_train_tgt), dtype="int")

X_train = np.concatenate([X_train_src, X_train_tgt])
y_train = np.concatenate([y_train_src, y_train_tgt])
d_train = np.concatenate([d_train_src, d_train_tgt])

X_val_src = np.concatenate([sweep_src_val, neutral_src_val])
y_val_src = np.array([1]*Ns_val + [0]*Ns_val)
d_val_src = np.zeros(len(X_val_src), dtype="int")

X_val_tgt = np.concatenate([sweep_tgt_val, neutral_tgt_val])
y_val_tgt = np.array([-1]*len(X_val_tgt))
d_val_tgt = np.ones(len(X_val_tgt), dtype="int")

X_val = np.concatenate([X_val_src, X_val_tgt])
y_val = np.concatenate([y_val_src, y_val_tgt])
d_val = np.concatenate([d_val_src, d_val_tgt])

X_test = np.concatenate([sweep_tgt_test, neutral_tgt_test])
y_test = np.array([1]*len(sweep_tgt_test) + [0]*len(neutral_tgt_test))
d_test = np.ones(len(X_test), dtype="int")

# ============================================================
# Prepare model inputs
# ============================================================
label_mask_train = (y_train != -1).astype("float32")
label_mask_val = (y_val != -1).astype("float32")

y_train_use = np.where(y_train == -1, 0, y_train).astype("float32").reshape(-1, 1)
y_val_use = np.where(y_val == -1, 0, y_val).astype("float32").reshape(-1, 1)
y_test_use = y_test.astype("float32").reshape(-1, 1)

d_train_onehot = keras.utils.to_categorical(d_train, 2)
d_val_onehot = keras.utils.to_categorical(d_val, 2)
d_test_onehot = keras.utils.to_categorical(d_test, 2)

X_train = X_train.astype("float32")
X_val = X_val.astype("float32")
X_test = X_test.astype("float32")

# ============================================================
# Build + compile model
# ============================================================
model, grl_layer = build_dann(input_shape=input_shape, alpha=1.0)

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss={"label": "binary_crossentropy", "domain": "categorical_crossentropy"},
    loss_weights={"label": 1.0, "domain": 1.0},
    metrics={"label": "accuracy", "domain": "accuracy"}
)

# ============================================================
# Training
# ============================================================
epochs = args.epochs
batch = args.batch

for epoch in range(epochs):
    p = epoch / epochs
    alpha = 2.0 / (1.0 + np.exp(-10*p)) - 1.0
    grl_layer.set_alpha(alpha)

    model.fit(
        X_train,
        [y_train_use, d_train_onehot],
        sample_weight=[label_mask_train, np.ones(len(X_train))],
        batch_size=batch,
        epochs=1,
        verbose=1
    )

    # Validation stats
    label_pred_src, _ = model.predict(X_val_src, verbose=0)
    acc_src = np.mean((label_pred_src.flatten() > 0.5) == y_val_src)

    _, dom_pred = model.predict(X_val, verbose=0)
    acc_dom = np.mean(np.argmax(dom_pred, axis=1) == d_val)

    print(f"EPOCH {epoch+1}/{epochs}  α={alpha:.3f}  SRC_ACC={acc_src:.4f}  DOM_ACC={acc_dom:.4f}")

# ============================================================
# Test Evaluation
# ============================================================
test_results = model.evaluate(X_test, [y_test_use, d_test_onehot], verbose=0)
print("\nFINAL TEST:")
print(f"Label Loss: {test_results[1]:.4f}")
print(f"Label Acc:  {test_results[3]:.4f}")
print(f"Domain Acc: {test_results[4]:.4f}")

label_pred, domain_pred = model.predict(X_test)
np.savetxt("class_pred.txt", label_pred)
np.savetxt("domain_pred.txt", domain_pred)

print("\nSaved predictions to class_pred.txt & domain_pred.txt")
