import numpy as np
import tensorflow as tf
import transformers
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

model_name = "cl-tohoku/bert-base-japanese"
tokenizer = transformers.BertTokenizer.from_pretrained(model_name)

data = pd.read_csv('ダミー.csv',sep=',',usecols=[1,2,3],nrows=500)
data = data.dropna(how='any')
data['labels'] = data['labels'].astype(dtype=int)
print(data)
train_text,test_text = train_test_split(data, test_size=0.2)

# 訓練データ
train_texts = train_text['Q22']
train_labels = train_text['labels']
print(train_labels)

# テストデータ
test_texts = test_text['Q22']
test_labels = test_text['labels']
print(test_labels)

# テキストのリストをtransformers用の入力データに変換
def to_features(texts, max_length):
    shape = (len(texts), max_length)
    input_ids = np.zeros(shape, dtype="int32")
    attention_mask = np.zeros(shape, dtype="int32")
    token_type_ids = np.zeros(shape, dtype="int32")
    for i, text in enumerate(texts):
        encoded_dict = tokenizer.encode_plus(text, max_length=max_length, pad_to_max_length=True)
        input_ids[i] = encoded_dict["input_ids"]
        attention_mask[i] = encoded_dict["attention_mask"]
        token_type_ids[i] = encoded_dict["token_type_ids"]
    return [input_ids, attention_mask, token_type_ids]

# 単一テキストをクラス分類するモデルの構築
def build_model(model_name, num_classes, max_length):
    input_shape = (max_length, )
    input_ids = tf.keras.layers.Input(input_shape, dtype=tf.int32)
    attention_mask = tf.keras.layers.Input(input_shape, dtype=tf.int32)
    token_type_ids = tf.keras.layers.Input(input_shape, dtype=tf.int32)
    bert_model = transformers.TFBertModel.from_pretrained(model_name)
    last_hidden_state, pooler_output = bert_model(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids
    )
    output = tf.keras.layers.Dense(num_classes, activation="softmax")(pooler_output)
    model = tf.keras.Model(inputs=[input_ids, attention_mask, token_type_ids], outputs=[output])
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["acc"])
    return model

num_classes = 3
max_length = 200
batch_size = 10
epochs = 3

x_train = to_features(train_texts, max_length)
y_train = tf.keras.utils.to_categorical(train_labels, num_classes=num_classes)
model = build_model(model_name, num_classes=num_classes, max_length=max_length)

def plot_accuracy_and_loss(history):
    print(history.history)
    accuracy = history.history['acc']
    plt.plot(accuracy, label='Accuracy against Training Data')
    plt.legend()
    plt.figure()
    plt.show()

    loss = history.history['loss']
    plt.plot(loss, label='Loss against Training Data')
    plt.legend()
    plt.figure()
    plt.show()

# 訓練
history = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs
)

# 予測
x_test = to_features(test_texts, max_length)
y_test = np.asarray(test_labels)
y_preda = model.predict(x_test)
y_pred = np.argmax(y_preda, axis=1)

model.summary()
model.save_weights('Emotions_Q22_model.h5',overwrite=True,save_format=None,options=None)

plot_accuracy_and_loss(history)