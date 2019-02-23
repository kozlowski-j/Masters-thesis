import pandas as pd
import numpy as np
# import openpyxl
import time

class Rob():
    '''
    class with robust functions
    '''
    def clean_data(self, df):
        '''
        data cleaninig
        -give beers correct id's
        -remove repeated reviews (when user reviews the same beer more than once)
        -remove beers with less than 10 reviews
        -remove users with less than 20 reviews
        -leave only useful columns:
        ['review_profilename', 'beer_id', 'beer_name', 'review_overall', 'user_review_count']
        :return: dataframe
        '''
        start = time.time()
        # rewriting beer id's because after Tableau Prep cleaning
        # one beer name can have several id numbers
        tmp = pd.DataFrame(df.groupby(['beer_name'])['beer_beerid'].count())
        tmp['beer_id'] = [i for i in range(0, tmp.shape[0])]
        tmp['beer_name'] = tmp.index
        tmp = tmp.reset_index(drop=True)
        df2 = df.merge(tmp[['beer_name', 'beer_id']], on='beer_name')

        # remove repeated reviews (when user reviews the same beer more than once
        # review_overall is taken as mean out of all the repeated reviews
        df2['dups'] = df2.groupby(['review_profilename', 'beer_name'])['beer_id'].transform('count')
        df2['mean_overall'] = df2.groupby(['review_profilename', 'beer_name'])['review_overall'].transform('mean')
        df2['review_overall'] = np.where(df2['dups'] > 1,
                                        df2['mean_overall'],
                                        df2['review_overall'])
        df2.drop_duplicates(['review_profilename', 'beer_name'], inplace=True)
        df2.drop(columns=['dups', 'mean_overall'], inplace=True)

        df2['beer_review_count'] = df2.groupby('beer_name')['review_overall'].transform('count')
        df2 = df2[df2['beer_review_count'] > 9]
        df2['user_review_count'] = df2.groupby('review_profilename')['review_overall'].transform('count')
        df2 = df2[df2['user_review_count'] > 19]

        df2 = df2[['review_profilename', 'beer_id', 'beer_name', 'review_overall', 'user_review_count']]
        print("clean_data() -", round(time.time() - start, 2), "s")
        return df2

    def descriptive(self, df):
        '''
        Creates descriptive analysis for columns in a given df.
        Useful during data preparation.
        :param df:
        :return: desc_df
        '''
        start = time.time()
        desc = []
        for column in df.select_dtypes('O').columns:
            desc_current = {}
            desc_current.update({'column': column,
                                 'size': df[column].size,
                                 'unique': df[column].unique().size})
            desc.append(desc_current)
        for column in df.select_dtypes('float64').columns:
            desc_current = {}
            desc_current.update({'column': column,
                                 'size': df[column].size,
                                 'min': df[column].min(),
                                 'mean': df[column].mean(),
                                 'median': df[column].median(),
                                 'max': df[column].max(),
                                 'std': df[column].std(),
                                 'skewness': df[column].skew(),
                                 'kurtosis': df[column].kurtosis()
                                 })
            desc.append(desc_current)

        print("descriptive() -", round(time.time() - start), "s")
        return pd.DataFrame(desc)

    def prep_data_format(self, df):
        '''
        Creates pivot table with binary values (instead of review_overall values).

        :param df:
        :return: pivot_binary, pivot_df
        '''
        start = time.time()
        # data format for apriori algorithm
        pivot_df = df.pivot(index='review_profilename',
                            columns='beer_id',
                            values='review_overall')
        pivot_df.fillna(0, inplace=True)
        pivot_binary = pivot_df.applymap(lambda x: 1 if x > 0 else 0)

        # data format for FP-growth algorithm
        pivot_beer_ids = pivot_binary.copy()
        for col in pivot_beer_ids.columns:
            pivot_beer_ids[col] = pivot_beer_ids[col] * int(col)
        transactions = [[i for i in lst if i != 0] for lst in pivot_beer_ids.values]

        print("prep_data_format() -", round(time.time() - start), "s")
        return pivot_binary, transactions, pivot_df


    def create_crossval(self, df, k_folds):
        '''
        Transforms data frame into test set and train set.
        // Still needs rethinking and some work on it

        :param df: cleaned df
        :param k_folds: number of subsets for multiple cross-val
        :return: test_df, train_df
        '''
        t500 = df[df['user_review_count'] > 500]['review_profilename'].unique()

        cv = {}
        t500_df = df[df['review_profilename'].isin(t500) == True].copy()
        _cv = t500_df.copy()
        for k in range(0, k_folds):
            cv[k] = _cv.groupby('review_profilename', group_keys=False).apply(
                lambda x: x.sample(int(1/k_folds * x['user_review_count'].mean())))
            cv_list = cv[k][['review_profilename', 'beer_id']]
            _cv = _cv[_cv.isin(cv_list) == False].dropna()

        test_df = cv[k]
        train_df = df[df.isin(test_df[['review_profilename', 'beer_id']]) == False].dropna()

        return test_df, train_df

    def spr_piwo(self, beer_id, df, df2, rule_cons):
        beer_name = df2[df2['beer_id'] == beer_id]['beer_name'].unique()[0]
        brewery_name = df[df['beer_name'] == beer_name]['brewery_name'].unique()[0]
        beer_style = df[df['beer_name'] == beer_name][ 'beer_style'].unique()[0]
        beer_abv = df[df['beer_name'] == beer_name]['beer_abv'].unique()[0]
        avg_rating = round(df[df['beer_name'] == beer_name]['review_overall'].mean(), 2)
        rev_cnt = df[df['beer_name'] == beer_name].index.size
        apprs_cnt = rule_cons[rule_cons['consequents'].isin([beer_id])]['Rule'].unique().size
        apprs_cnt_prc = rule_cons['Rule'].unique().size
        print("Beer info:\n"
              "- beer name: {}\n"
              "- brewery name: {}\n"
              "- beer style: {}\n"
              "- beer abv: {}\n"
              "- number of reviews: {}\n"
              "- average rating: {}\n"
              "- appears as consequent in {} ({}%) rules\n".format(beer_name, brewery_name, beer_style,
                                                             beer_abv, rev_cnt, avg_rating, apprs_cnt, apprs_cnt_prc))

    def limit_reviews(self, df, review_overall_cutoff_value):
        return df[df['review_overall'] >= review_overall_cutoff_value]









