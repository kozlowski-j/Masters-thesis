import pandas as pd
import numpy as np
from robust import Rob
from fim import apriori, eclat, fpgrowth, fim, sam, relim
import time


class MBA_fim():
    '''
    Market Basket Analysis - apriori
    '''

    def mbasket(self, data_p, support_par, confidence_par, method='apriori'):
        """
        :param
        :return:
        """
        start = time.time()
        ## Apriori analysis + association rules creation
        # find association rules with default settings
        lift_par = 1.2
        if method == 'apriori':
            rules = pd.DataFrame(apriori(data_p[1], supp=support_par*100, conf=confidence_par*100,
                                         zmin=1, target='r', report='sclxy', mode='o', eval='l',
                                         thresh=1.2),
                                 columns=['consequents', 'antecedents', 'support',
                                          'confidence', 'lift', 'antecedent_support', 'consequent_support'])
        if method == 'fpgrowth':
            rules = pd.DataFrame(fpgrowth(data_p[1], supp=support_par*100, conf=confidence_par*100,
                                          zmin=1, target='r', report='sclxy', mode='o', eval='l',
                                          thresh=1.2),
                                 columns=['consequents', 'antecedents', 'support',
                                          'confidence', 'lift', 'antecedent_support', 'consequent_support'])
        if method == 'eclat':
            rules = pd.DataFrame(eclat(data_p[1], supp=support_par*100, conf=confidence_par*100,
                                       zmin=1, target='r', report='sclxy', mode='o', eval='l',
                                       thresh=1.2),
                                 columns=['consequents', 'antecedents', 'support',
                                          'confidence', 'lift', 'antecedent_support', 'consequent_support'])
        # if method == 'relim':
        #     rules = pd.DataFrame(sam(data_p[1], supp=support_par*100,
        #                                zmin=1, target='r', report='sclxy', mode='o', eval='l',
        #                                thresh=1.2),
        #                          columns=['consequents', 'antecedents', 'support',
        #                                   'confidence', 'lift', 'antecedent_support', 'consequent_support'])

        # ## Add column with count of antecedents and consequents for each rule
        # rules["antecedent_len"] = rules["antecedents"].apply(lambda x: x.__len__())
        #
        # ## Filter rules by parametres
        # rules = rules[(rules['lift'] >= lift_par) &
        #               (rules['confidence'] >= confidence_par) &
        #               (rules['antecedent_len'] <= 10)]

        # users with antedecents from the rules calculated above
        if rules.shape[0] > 0 :
            pivot_binary_tr = data_p[0].transpose()
            recom = {}
            pb = {}
            rules['antecedents'] = rules['antecedents'].apply(lambda x: frozenset(x))
            for user in pivot_binary_tr.columns:
                products_bought = pivot_binary_tr.index[pivot_binary_tr[user] == 1]
                pb[user] = products_bought
                suitable_rules = []
                for ante in rules['antecedents'].iteritems():
                    if ante[1].issubset(products_bought): # do poprawy
                        suitable_rules.append(ante[0])
                recom[user] = suitable_rules

            recom = pd.DataFrame.from_dict(recom, orient='index').stack().reset_index(level=1, drop=True).reset_index()
            recom.columns = ['review_profilename', 'Rule']

            # products bought - zeby wykluczyc te produkty z rekomendacji
            pb2 = pd.DataFrame.from_dict(pb, orient='index').stack().reset_index(level=1, drop=True).reset_index()
            pb2.columns = ['review_profilename', 'antecedents1']

            rule_cons = rules[['antecedents', 'consequents']].reset_index()
            # rule_cons['consequents'] = [i for i, *_ in rule_cons['consequents']]  # change format from frozensets to normal
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
        else:
            rule_cons = 0
            recom_new = 0
            sum_recom_already_satisfied = 0

        mba_time = round(time.time() - start)
        print("mbasket() -", mba_time, "s")

        return [rule_cons, recom_new, mba_time, sum_recom_already_satisfied]




