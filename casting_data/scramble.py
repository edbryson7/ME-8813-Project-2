import pandas as pd

df = pd.read_csv('test.csv')
print(df)
df = df.sample(frac=1)
print(df)
