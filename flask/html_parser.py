from bs4 import BeautifulSoup
import file

html = file.html_table
soup = BeautifulSoup(html, 'html.parser')

cols_th = soup.find_all('th')
cols_td = soup.find_all('td')

# HTMLタグ、中身の文字列の取り出し
th_string_data = []
td_string_data = []
th_tag_data = []
td_tag_data = []
for col in cols_th:
    th_string_data.append(col.string)
    th_tag_data.append(col.unwrap())
for col in cols_td:
    td_string_data.append(col.string)
    td_tag_data.append(col.unwrap())

th_data = th_tag_data
td_data = td_tag_data

# <th>タグにclassの挿入
for col in range(len(cols_th)):
    th_data[col].attrs['class'] = "{sorter:'metadata'}"
    th_data[col].string = th_string_data[col]

# <td>タグにclassの挿入
counter = 0
for col in range(len(cols_td)):
    # 回答の文字数の値を挿入
    if col == 1 + 5 * counter:
        td_data[col].attrs['class'] = '{sortValue: ' + format(len(td_string_data[col])) + '}'
    # %の値を挿入
    if col == 2 + 5 * counter or col == 3 + 5 * counter or col == 4 + 5 * counter:
        td_data[col].attrs['class'] = '{sortValue: ' + td_string_data[col] + '}'
    # カウントのインクリメント
    if col == 4 + 5 * counter:
        counter +=1
    td_data[col].string = td_string_data[col]


# <tr>,</td>タグの挿入
counter = 0
for col in range(len(cols_td)):
    if col == 5 * counter:
        td_data[col] = '<tr class="' + td_string_data[col] + '">' + '' + str(td_data[col])
    if col == 4 + 5 * counter:
        td_data[col] = str(td_data[col]) + '' + '</tr>'
        counter += 1


# <th>, <td>の内容を結合
th_string = '\n'.join(map(str, th_data))
td_string = '\n'.join(map(str, td_data))

# BeautifulSoupのobject化
th_soup = BeautifulSoup(th_string, 'html.parser')
td_soup = BeautifulSoup(td_string, 'html.parser')

# 元のHTMLに挿入
soup.thead.clear()
soup.thead.insert(0, th_soup)
soup.tbody.clear()
soup.tbody.insert(0, td_soup)
#print(soup)

