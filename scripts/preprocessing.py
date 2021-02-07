import pandas as pd
import numpy as np


# Module to pre-process the Starbucks datasets

def read_data(file_path, file_type='json'):
    if file_type == 'json':
        df = pd.read_json(file_path, orient='records', lines=True)
    elif file_type == 'csv':
        df = pd.read_csv(file_path)
    return df


def preprocess_data(df, data_name):
    if data_name == 'portfolio':
        df_explode = df.explode('channels')
        df_explode['_helper'] = 1
        df_expanded = df_explode.pivot(index='id', columns='channels', values='_helper')
        df_expanded.fillna(0, inplace=True)
        df_preprocessed = df.merge(df_expanded, how='left', left_on='id', right_index=True)
        df_preprocessed["offer name"] = ['Offer {}'.format(i + 1) for i in range(len(df_preprocessed))]
    elif data_name == 'profile':
        df_preprocessed = df.copy()
        df_preprocessed['date_joined'] = pd.to_datetime(df_preprocessed['became_member_on'], format='%Y%m%d')
        df_preprocessed['year_joined'] = df_preprocessed['date_joined'].dt.year
        df_preprocessed.loc[df_preprocessed['age'] == 118, 'age'] = np.nan
    elif data_name == 'transcript':
        print('transcript data preprocessing not supported')
        pass
    else:
        raise ValueError('data_name is not one of "portfolio", "profile", or "transcript"')
    return df_preprocessed


if __name__ == '__main__':
    portfolio = pd.read_json('../data/portfolio.json', orient='records', lines=True)
    profile = pd.read_json('../data/profile.json', orient='records', lines=True)
    transcript = pd.read_json('../data/transcript.json', orient='records', lines=True)
    portfolio_pp = preprocess_data(portfolio, data_name='portfolio')
    print(portfolio_pp.iloc[0])
