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
    data_p = rob.prep_data_format(df_rdy)

    recom, rules = mba.mbasket(data_p, 0.1, 'ap')
    recom2, rules2 = mba.mbasket(data_p, 0.1, 'fp')

    # for col in recom.columns:
    #     if col != 'antecedents':
    #         print(recom[col].value_counts().head(10))
    print(rules.head(), "\n_________________\n")
    print(rules2.head(), "\n_________________\n")
    print(recom.head(), "\n_________________\n")
    print(recom2.head())
    # exit()