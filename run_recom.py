import pandas as pd
import numpy as np
from robust import Rob
from mba2 import MBA


if __name__ == '__main__':
    rob = Rob()
    mba = MBA()
    df = pd.read_pickle('beer_reviews_complete.pkl')
    df2 = rob.clean_data(df)
    df_rdy = rob.limit_reviews(df2, 4)
    cv = rob.create_crossval(df_rdy, 10)
    comp_tab = []
    for k in range(0, 2):
        test_df = cv[k].copy()
        train_df = df_rdy[df_rdy.isin(test_df[['review_profilename', 'beer_id']]) == False].dropna()
        data_p = rob.prep_data_format(train_df)

        rules, recom, run_time, sum_recom_already_satisfied = mba.mbasket(data_p, 0.08, 'ap')
        comp_tab.append(rob.comparer(recom, test_df, k, 'ap', run_time, sum_recom_already_satisfied))

        rules2, recom2, run_time2, sum_recom_already_satisfied2 = mba.mbasket(data_p, 0.08, 'fp')
        comp_tab.append(rob.comparer(recom2, test_df, k, 'fp', run_time2, sum_recom_already_satisfied2))

    print(pd.DataFrame(comp_tab))

    # print(rules.head(), "\n_________________\n")
    # print(rules2.head(), "\n_________________\n")
    # print(recom.head(), "\n_________________\n")
    # print(recom2.head())
