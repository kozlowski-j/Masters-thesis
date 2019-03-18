import pandas as pd
import numpy as np
from robust import Rob
from mba2 import MBA
import openpyxl

if __name__ == '__main__':
    rob = Rob()
    mba = MBA()
    df = pd.read_pickle('beer_reviews_complete.pkl')
    df2 = rob.clean_data(df)
    df_rdy = rob.limit_reviews(df2, 4)
    cv = rob.create_crossval(df_rdy, 10)
    comp_tab = []
    for k in range(0, 10):
        test_df = cv[k].copy()
        train_df = df_rdy[df_rdy.isin(test_df[['review_profilename', 'beer_id']]) == False].dropna()
        data_p = rob.prep_data_format(train_df)
        for support_par in [(i + 1) / 1000 for i in range(70, 150)]:
            for confidence_par in [(i + 1) / 100 for i in range(50, 90)]:

                results = mba.mbasket(data_p, support_par, confidence_par, 'ap')
                comp_tab.append(rob.comparer(test_df, k, 'ap', support_par,
                                             confidence_par, results))

                results = mba.mbasket(data_p, support_par, confidence_par, 'fp')
                comp_tab.append(rob.comparer(test_df, k, 'fp', support_par,
                                             confidence_par, results))

    print(pd.DataFrame(comp_tab))
    comp = pd.DataFrame(comp_tab)
    print("Total MBA time: {} s".format(comp['mba time [s]'].sum()))

    comp.to_excel('comp_test009-01.xlsx')

    # print(rules.head(), "\n_________________\n")
    # print(rules2.head(), "\n_________________\n")
    # print(recom.head(), "\n_________________\n")
    # print(recom2.head())
