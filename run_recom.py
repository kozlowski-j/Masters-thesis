import pandas as pd
from robust import Rob
from mba_fim import MBA_fim
import openpyxl

if __name__ == '__main__':
    rob = Rob()
    mba = MBA_fim()
    df = pd.read_pickle('beer_reviews_complete.pkl')
    df2 = rob.limit_reviews(df, 4)
    df_rdy = rob.clean_data(df2)
    cv = rob.create_crossval(df_rdy, 5)
    comp_tab = []
    algos_list = ['apriori', 'fpgrowth', 'eclat', 'relim', 'sam']
    lift_par = 1.2
    for k in range(0, 5):
        print("==================== k:", k, "====================")
        test_df = cv[k].copy()
        train_df = df_rdy[df_rdy.isin(test_df[['review_profilename', 'beer_id']]) == False].dropna()
        data_p = rob.prep_data_format(train_df)
        for support_par in [i / 1000 for i in range(90, 130, 5)]:
            for confidence_par in [i / 100 for i in range(70, 85, 5)]:
                for algorithm in algos_list:
                    print(k, "==================== ", algorithm, "support:", support_par,
                          "confidence_par:", confidence_par, "====================")
                    results = mba.mbasket(data_p, support_par, confidence_par, algorithm, lift_par)
                    comp_tab.append(rob.comparer(test_df, k, algorithm, support_par,
                                                 confidence_par, results, lift_par))
            comp_tmp = pd.DataFrame(comp_tab).to_excel('comp_tmp.xlsx')
            print("====================\nTmp file saved.\n====================")

    print(pd.DataFrame(comp_tab))
    comp = pd.DataFrame(comp_tab)
    print("Total MBA time: {} s".format(comp['mba time [s]'].sum()))
    comp.to_excel('comp_popr0.xlsx')

# # porysowac jak najwiecej wykresow do rozdzialu z porownaniem
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# sns.jointplot(x=rules['confidence'], y=rules['lift'])
# plt.show()
