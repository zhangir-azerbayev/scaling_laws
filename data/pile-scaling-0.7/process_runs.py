import pickle
import pandas as pd

PARAM_KEY = {
    "70M": 70_426_624,
    "160M": 162_322_944,
    "410M": 405_334_016,
    "1-4B": 1_414_647_808,
    "2-8B": 2_775_208_960,
    "6-9B": 6_857_302_016,
    "12B": 11_846_072_320
}

BATCH_SIZE = 2**21

def main():
    df = pd.read_csv('pile-scaling-0.7.csv')

    cols_to_drop = [col for col in df.columns if 'MIN' in col or 'MAX' in col]

    df = df.drop(columns=cols_to_drop)

    df = df.sum(axis=0, skipna=True)

    runs = []

    for name, loss in df.iteritems():
        if 'Group' in name:
            tokenized = name.replace('Group: ', '').split('_')
            params = PARAM_KEY[tokenized[0]]
            tokens = BATCH_SIZE * int(tokenized[1].replace('step', ''))
            
            runs.append(dict(N=params, D=tokens, L=loss))

    with open('runs.pkl', 'wb') as fle:
        pickle.dump(runs, fle)


if __name__=="__main__":
    main()
