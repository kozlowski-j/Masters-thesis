import pandas as pd
import numpy as np
from robust import Rob
from mlxtend.frequent_patterns import apriori, association_rules


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
        support_par = 0.8
        lift_par = 1.2
        confidence_par = 0.7
        ## Apriori analysis + association rules creation
        # find association rules with default settings
        support_par = min(support_par, 2000 / pivot_binary.shape[0])
        frequent_itemsets = apriori(pivot_binary, min_support=support_par, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

        ## Add column with count of antecedents and consequents for each rule
        rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
        rules["consequent_len"] = rules["consequents"].apply(lambda x: len(x))

        ## Filter rules by parametres
        rules = rules[(rules['lift'] >= lift_par) &
                      (rules['confidence'] >= confidence_par) &
                      (rules['antecedent_len'] <= antecedent_length) &
                      (rules['consequent_len'] == 1)]


        # users with antedecents from the rules calculated above
        pivot_binary_tr = pivot_binary.transpose()
        recom = {}
        pb = {}
        for user in pivot_binary_tr:
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

        rule_cons = rules['consequents'].reset_index()
        rule_cons['consequents'] = [i for i, *_ in rule_cons['consequents']]  # change format from frozensets to normal
        rule_cons.columns = ['Rule', 'consequents']
        recom = recom.merge(rule_cons, on='Rule')

        recom = recom[recom.isin(pb2)==False].dropna()

        return rules




if __name__ == '__main__':
    rob = Rob()
    df = pd.read_pickle('beer_reviews.pkl')
    df = rob.clean_data(df)
    pivot_binary = rob.pivots(df)[0]

    print(df.head())

    exit()


