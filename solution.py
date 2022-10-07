import pandas as pd

if __name__ == '__main__':
    train = pd.read_csv("train.csv", index_col='Id')
    test = pd.read_csv("test.csv", index_col='Id')
    test['y'] = test.mean(axis=1)
    test[['y']].to_csv('output.csv')
