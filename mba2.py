import pandas as pd
import numpy as np
from robust import Rob
from mlxtend.frequent_patterns import apriori, association_rules
import time


class MBA():
    '''
    Market Basket Analysis
    '''

    antecedent_length = 5  # how deep to drill

    def rule_param_adjust(self, support, confidenct, lift):
        # arbitrary coefficients adjusting the rule parameteres in order to choose the best rule for user
        # consider to use LEVERAGE or CONVICTION from mlextend instead
        return 0.4 * support + 0.5 * confidenct + 0.1 * lift

    def recommendation_choice(self, recommendations):
        # for each cust/prod sort recommendations by id, adjusted params and antedecent length. Take the first only.
        recommendations = recommendations.sort_values(['ids', 'm', 'antecedent_len'], ascending=False)
        return recommendations[~recommendations['ids'].duplicated()]

    def from_frozenset(self, series):
        # change format from frozensets to normal
        return [i for i, *_ in series]

    def how_many_output_antecedents(self, rules_columns):
        # this is to avoid exceptions in legacy formatting
        pattern = re.compile('^lhs')
        return len([pattern.findall(i) for i in rules_columns if len(pattern.findall(i)) > 0])


    def mbasket(self, pivot_binary, min_potential_value):
        """
        :param
        :return:
        """

        ## Apriori analysis + association rules creation
        # find association rules with default settings
        # support_par = min(support_par, 2000 / pivot_binary.shape[0])
        support_par = 0.12
        lift_par = 1.2
        confidence_par = 0.6
        frequent_itemsets = apriori(pivot_binary, min_support=support_par, use_colnames=True)
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
        pivot_binary_tr = pivot_binary.transpose()
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
        pb2.columns = ['review_profilename', 'antecedents']

        rule_cons = rules[['antecedents', 'consequents']].reset_index()
        rule_cons['consequents'] = [i for i, *_ in rule_cons['consequents']]  # change format from frozensets to normal
        rule_cons['antecedents'] = [list(i) for i in rule_cons['antecedents']]
        rule_cons.columns = ['Rule', 'antecedents', 'consequents']
        recom = recom.merge(rule_cons, on='Rule')

        recom = np.where()


        # recom = recom[recom.isin(pb2) == False].dropna()

        return rules




if __name__ == '__main__':
    rob = Rob()
    # df = pd.read_pickle('beer_reviews.pkl')
    # df = pd.read_csv(r"C:\Users\jp_ko\OneDrive\Studia\SGH\Magisterka\beer_reviews.csv")
    # df.to_pickle('beer_reviews_complete.pkl')
    df = pd.read_pickle('beer_reviews_complete.pkl')
    df2 = rob.clean_data(df)
    df_desc = rob.descriptive(df)

    pivot_binary = rob.pivots(df2)[0]

    print(df.head())

    exit()


