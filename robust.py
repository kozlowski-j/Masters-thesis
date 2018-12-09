import pandas as pd
import numpy as np
# import openpyxl

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
        return df2

    def descriptive(self, df):
        '''
        Creates descriptive analysis for columns in a given df.
        Useful during data preparation.
        :param df:
        :return: desc_df
        '''
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
        return pd.DataFrame(desc)

    def pivots(self, df):
        '''
        Creates pivot table with binary values (instead of review_overall values).

        :param df:
        :return: pivot_binary, pivot_df
        '''
        pivot_df = df.pivot(index='review_profilename',
                            columns='beer_id',
                            values='review_overall')
        pivot_df.fillna(0, inplace=True)

        pivot_binary = pivot_df.applymap(lambda x: 1 if x > 0 else 0)

        return pivot_binary, pivot_df

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

    # def run(self):
    #     #histogram browarow
    #     df['brewery_name'].value_counts()[np.abs(df['brewery_name'].value_counts() - \
    #                                                    df['brewery_name'].value_counts().mean()) <= \
    #                                             (2*df['brewery_name'].value_counts().std())].hist()
    #     # plt.show()
    #
    #
    #     df['review_profilename'].value_counts()[np.abs(df['review_profilename'].value_counts() - \
    #                                                    df['review_profilename'].value_counts().mean()) <= \
    #                                             (2*df['review_profilename'].value_counts().std())].hist()
    #
    #
    #     writer = pd.ExcelWriter('descriptive_analytics.xlsx')
    #     desc_numeric.to_excel(writer, 'desc_numeric')
    #     desc_non_numeric.to_excel(writer, 'desc_non_numeric')
    #     writer.save()
    #     return

# rob = Rob()
# # data prep
# # df = pd.read_csv('beer_reviews_tableau_cleaned.csv', sep=';', decimal=',')
# df = pd.read_pickle('beer_reviews.pkl')
# df = rob.clean_data(df)
#
# test, train = rob.create_crossval(df, 5)
#
# pivot_binary = rob.pivots(train)[0]
#
# desc2 = rob.descriptive(df)







