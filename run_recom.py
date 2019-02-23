import pandas as pd
import numpy as np
from robust import Rob
from mba2 import MBA


if __name__ == '__main__':
    rob = Rob()
    mba = MBA()
    # df = pd.read_pickle('beer_reviews.pkl')
    # df = pd.read_csv(r"C:\Users\jp_ko\OneDrive\Studia\SGH\Magisterka\beer_reviews.csv")
    # df.to_pickle('beer_reviews_complete.pkl')
    df = pd.read_pickle('beer_reviews_complete.pkl')
    df2 = rob.clean_data(df)
    df_rdy = rob.limit_reviews(df2, 4)
    # print(rob.descriptive(df))
    pivot_binary = rob.pivots(df_rdy)[0]

    recom, rules = mba.mbasket(pivot_binary, 0.1)

    # for col in recom.columns:
    #     if col != 'antecedents':
    #         print(recom[col].value_counts().head(10))
    print(rules.head(), "\n_________________\n")
    print(recom.head())
    # exit()