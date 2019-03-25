import pandas as pd
import numpy as np
from robust import Rob
from mba2 import MBA
from mba_fim import MBA_fim
import openpyxl

if __name__ == '__main__':
    rob = Rob()
    mba = MBA_fim()
    df = pd.read_pickle('beer_reviews_complete.pkl')
    algos_list = ['apriori', 'fpgrowth', 'eclat']
    df2 = rob.clean_data(df)
    df_rdy = rob.limit_reviews(df2, 4)
    cv = rob.create_crossval(df_rdy, 10)
    comp_tab = []
    for k in range(0, 5):
        print("==================== k:", k, "====================")
        test_df = cv[k].copy()
        train_df = df_rdy[df_rdy.isin(test_df[['review_profilename', 'beer_id']]) == False].dropna()
        data_p = rob.prep_data_format(train_df)
        for support_par in [i / 1000 for i in range(80, 120, 5)]:
            for confidence_par in [i / 100 for i in range(60, 90, 5)]:
                for algorithm in algos_list:
                    print(k, "==================== ", algorithm, "support:", support_par,
                          "confidence:", confidence_par, "====================")
                    results = mba.mbasket(data_p, support_par, confidence_par, algorithm)
                    comp_tab.append(rob.comparer(test_df, k, algorithm, support_par,
                                                 confidence_par, results))

                comp_tmp = pd.DataFrame(comp_tab).to_excel('comp_tmp.xlsx')
                print("====================\nTmp file saved.\n====================")

    print(pd.DataFrame(comp_tab))
    comp = pd.DataFrame(comp_tab)
    print("Total MBA time: {} s".format(comp['mba time [s]'].sum()))
    comp.to_excel('comp_fim_008-012.xlsx')

