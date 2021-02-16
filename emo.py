import numpy as np
import tensorflow as tf
import transformers
import csv
import pandas as pd

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

model_name = "cl-tohoku/bert-base-japanese"
tokenizer = transformers.BertTokenizer.from_pretrained(model_name)
num_classes = 3
max_length = 200
model = build_model(model_name, num_classes=num_classes, max_length=max_length)
model.load_weights('Emotions_Q22_model.h5')
file = 'ダミー.csv'

f = open(file,'r')
data = []
for row in csv.reader(f):
    data.append(row[2])
f.close()

del data[0]
print(data)
text = data

def to_features(texts, max_length):
    shape = (len(texts), max_length)
    # input_idsやattention_mask, token_type_idsの説明はglossaryに記載(cf. https://huggingface.co/transformers/glossary.html)
    input_ids = np.zeros(shape, dtype="int32")
    attention_mask = np.zeros(shape, dtype="int32")
    token_type_ids = np.zeros(shape, dtype="int32")
    for i, text in enumerate(texts):
        encoded_dict = tokenizer.encode_plus(text, max_length=max_length, pad_to_max_length=True)
        input_ids[i] = encoded_dict["input_ids"]
        attention_mask[i] = encoded_dict["attention_mask"]
        token_type_ids[i] = encoded_dict["token_type_ids"]
    return [input_ids, attention_mask, token_type_ids]

x_test = to_features(text,max_length)
y_preda = model.predict(x_test)*100
print(y_preda)
y_pred = np.argmax(y_preda, axis=1)
print(y_pred)

np.savetxt('%_Q22.csv', y_preda, delimiter=',', fmt='%.5f')
np.savetxt('emo_Q22.csv', y_pred, delimiter=',', fmt='%d')
df1 = pd.read_csv('%_Q22.csv')
df2 = pd.read_csv('emo_Q22.csv')
df1.columns = ['ニュートラル', 'ポジティブ', 'ネガティブ']
df2.columns = ['予測値']
df_concat = pd.concat([df2, df1], axis=1)
df_concat.to_csv('dami-', index=None)





