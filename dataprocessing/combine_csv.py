import pandas as pd


DATA_DIR = "../data"
df1 = pd.read_csv('{}/data1.csv'.format(DATA_DIR))
df2 = pd.read_csv('{}/data2.csv'.format(DATA_DIR))
df3 = pd.read_csv('{}/data3.csv'.format(DATA_DIR))
df4 = pd.read_csv('{}/data4.csv'.format(DATA_DIR))
df5 = pd.read_csv('{}/data5.csv'.format(DATA_DIR))
df6 = pd.read_csv('{}/data6.csv'.format(DATA_DIR))
print(df1.size)
out = df1.append(df2)
print(out.size)
out = out.append(df3)
print(out.size)
out = out.append(df4)
print(out.size)
out = out.append(df5)
print(out.size)
out = out.append(df5)
print(out.size)

with open('{}/data.csv'.format(DATA_DIR), 'w', encoding='utf-8') as f:
    out.to_csv(f, index=False)