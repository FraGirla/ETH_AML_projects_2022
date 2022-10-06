import pandas as pd

if __name__ == '__main__':
    train = pd.read_csv("train.csv", index_col='Id')
    test = pd.read_csv("test.csv", index_col='Id')
    output = pd.DataFrame()
    test['y'] = test.mean(axis=1)
    #print(test[['y']])
    test[['y']].to_csv('output.csv')
