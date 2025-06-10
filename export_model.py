# export_model.py（修正版）
import tensorflow as tf
import numpy as np
import json
import os

os.makedirs("model", exist_ok=True)

# 載入 Fashion-MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0

# 建立模型
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    
    tf.keras.layers.Dense(10, activation='softmax')
])

# 訓練模型
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=40, validation_data=(x_test, y_test))

# 儲存架構與對應權重名稱（自定義名稱方便後續查找）
arch = []
weights_dict = {}

for i, layer in enumerate(model.layers):
    wnames = []
    for j, w in enumerate(layer.weights):
        name = f"layer{i}_w{j}"
        weights_dict[name] = w.numpy()
        wnames.append(name)
    
    arch.append({
        "name": layer.name,
        "type": layer.__class__.__name__,
        "config": layer.get_config(),
        "weights": wnames
    })

# 儲存 JSON 與 NPZ 檔案
with open("model/fashion_mnist.json", "w") as f:
    json.dump(arch, f)

np.savez("model/fashion_mnist.npz", **weights_dict)
