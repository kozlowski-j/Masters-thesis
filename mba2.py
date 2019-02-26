import pandas as pd
import numpy as np
from robust import Rob
from mlxtend.frequent_patterns import apriori, association_rules
from fp_growth3 import find_frequent_itemsets
from eclat import eclat
import time


class MBA():
    '''
    Market Basket Analysis - apriori
    '''

    # def find_freq_itemsets_eclat(self, transactions, support_par):
    #
    #     return apriori(pivot_binary, min_support=support_par, use_colnames=True)

    def find_freq_itemsets_apriori(self, pivot_binary, support_par):
        return apriori(pivot_binary, min_support=support_par, use_colnames=True)

    def find_freq_itemsets_fp_growth(self, transactions, support_par):

        min_support = round(support_par * len(transactions))

        fp_sets = find_frequent_itemsets(transactions, min_support)
        itsets = pd.DataFrame(fp_sets,
                              columns=['itemsets', 'support'])
        itsets['support'] = round(itsets['support'] / len(transactions), 5)
        return itsets

    def mbasket(self, data_p, support_par, method='ap'):
        """
        :param
        :return:
        """
        start = time.time()
        ## Apriori analysis + association rules creation
        # find association rules with default settings
        lift_par = 1.2
        confidence_par = 0.6
        if method == 'ap':
            start = time.time()
            frequent_itemsets = self.find_freq_itemsets_apriori(data_p[0], support_par)
            run_time = round(time.time() - start)
            print("find_freq_itemsets_apriori() -", run_time, "s")
        if method == 'fp':
            start = time.time()
            frequent_itemsets = self.find_freq_itemsets_fp_growth(data_p[1], support_par)
            run_time = round(time.time() - start)
            print("find_freq_itemsets_fp_growth() -", run_time, "s")

        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

        ## Add column with count of antecedents and consequents for each rule
        rules["antecedent_len"] = rules["antecedents"].apply(lambda x: x.__len__())
        rules["consequent_len"] = rules["consequents"].apply(lambda x: x.__len__())

        ## Filter rules by parametres
        rules = rules[(rules['lift'] >= lift_par) &
                      (rules['confidence'] >= confidence_par) &
                      (rules['antecedent_len'] <= 10) &
                      (rules['antecedent_len'] >= 2) &
                      (rules['consequent_len'] == 1)]

        # users with antedecents from the rules calculated above
        pivot_binary_tr = data_p[0].transpose()
        recom = {}
        pb = {}
        for user in pivot_binary_tr.columns:
            products_bought = pivot_binary_tr.index[pivot_binary_tr[user] == 1]
            pb[user] = products_bought
            suitable_rules = []
            for ante in rules['antecedents'].iteritems():
                if ante[1].issubset(products_bought):
                    suitable_rules.append(ante[0])
            recom[user] = suitable_rules

        recom = pd.DataFrame.from_dict(recom, orient='index').stack().reset_index(level=1, drop=True).reset_index()
        recom.columns = ['review_profilename', 'Rule']

        # products bought - zeby wykluczyc te produkty z rekomendacji
        pb2 = pd.DataFrame.from_dict(pb, orient='index').stack().reset_index(level=1, drop=True).reset_index()
        pb2.columns = ['review_profilename', 'antecedents1']

        rule_cons = rules[['antecedents', 'consequents']].reset_index()
        rule_cons['consequents'] = [i for i, *_ in rule_cons['consequents']]  # change format from frozensets to normal
        rule_cons['antecedents'] = [list(i) for i in rule_cons['antecedents']]
        rule_cons.columns = ['Rule', 'antecedents', 'consequents']
        recom = recom.merge(rule_cons, on='Rule')
        recom.drop_duplicates(['review_profilename', 'consequents'], keep='first', inplace=True)

        # exclude from recommendations products already bought
        recom_already_satisfied = pb2.merge(recom, left_on=['review_profilename', 'antecedents1'],
                                                   right_on=['review_profilename', 'consequents'])
        recom_already_satisfied['beer_already_known'] = 1
        sum_recom_already_satisfied = recom_already_satisfied['beer_already_known'].sum()

        recom_new = recom.merge(recom_already_satisfied[['review_profilename', 'Rule', 'consequents', 'beer_already_known']],
                                on=['review_profilename', 'Rule', 'consequents'],
                                how='left')
        recom_new = recom_new[recom_new['beer_already_known'] != 1][['review_profilename', 'Rule',
                                                                     'antecedents', 'consequents']]
        print("mbasket() -", round(time.time() - start), "s")
        return rule_cons, recom_new, run_time, sum_recom_already_satisfied




