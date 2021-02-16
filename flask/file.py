import pandas as pd

csv_file_path = 'ダミーデータ.csv'

df = pd.read_csv(csv_file_path)

# 各listにCSVの列項目を格納
className_list = list(df['className'])
answer_list = list(df['Q22'])
Neu_list = list(df['ニュートラル'])
Pos_list = list(df['ポジティブ'])
Neg_list = list(df['ネガティブ'])

# 科目の重複の削除
className_list_original = className_list
className_list = list(dict.fromkeys(className_list))

# "〇〇年度×学期　授業評価アンケート"の部分の削除
for index, item in enumerate(className_list):
    s = className_list[index].find('【')
    className_list[index] = className_list[index][s:]

for index, item in enumerate(className_list_original):
    s = className_list_original[index].find('【')
    className_list_original[index] = className_list_original[index][s:]

# %を削除
for index, item in enumerate(Neu_list):
    Neu_list[index] = Neu_list[index].rstrip('%')
    Pos_list[index] = Pos_list[index].rstrip('%')
    Neg_list[index] = Neg_list[index].rstrip('%')


# それぞれのパーセンテージ計算
Neu_count = 0
Pos_count = 0
Neg_count = 0
for index in range(len(Neu_list)):
    if Neu_list[index] > Pos_list[index] and Neu_list[index] > Neg_list[index]:
        Neu_count += 1
    elif Pos_list[index] > Neu_list[index] and Pos_list[index] > Neg_list[index]:
        Pos_count += 1
    elif Neg_list[index] > Neu_list[index] and Neg_list[index] > Pos_list[index]:
        Neg_count += 1

# listの作成
df_table = pd.DataFrame({'科目名': className_list_original,
                         '回答': answer_list,
                         'ポジティブ': Pos_list,
                         'ニュートラル': Neu_list,
                         'ネガティブ': Neg_list
                         })

html_table = df_table.to_html(classes='table', index=False, table_id='table_id')