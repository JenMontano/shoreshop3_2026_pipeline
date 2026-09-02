import pandas as pd
import numpy as np

def get_seasonality(index_dates, trend = 0 , periods = [2,4,6]):
        seasonality_df = pd.DataFrame()
        seasonality_df['time'] = index_dates

        seasonality_df['day_of_year_frac'] = (seasonality_df.time.dt.dayofyear - 1) / 365.25
        seasonality_df['year'] = seasonality_df.time.dt.year - seasonality_df.time.dt.year.min()
        seasonality_df['time_num'] = seasonality_df['year'] + seasonality_df['day_of_year_frac'] - seasonality_df['year'].values[0]

        seasonality_df = seasonality_df.drop(columns=['year','day_of_year_frac'])
        seasonality_df.set_index('time',inplace= True)

        for period in periods:
            seasonality_df[f'ss_sin_{period}'] = np.sin(period*np.pi*seasonality_df['time_num'])
            seasonality_df[f'ss_cos_{period}'] = np.cos(period*np.pi*seasonality_df['time_num'])

        if trend == 0:
            seasonality_df.drop(columns=['time_num'],inplace=True)

        elif trend == 1:
            seasonality_df.rename(columns={'time_num':'trend'}, inplace=True)

        elif trend == 2:
            seasonality_df['trend_squared'] = seasonality_df['time_num']**2
            seasonality_df.rename(columns={'time_num':'trend'}, inplace=True)
        else:
            raise ValueError("Invalid trend value. Must be 0, 1, or 2.")
        
        return seasonality_df

def get_covariates(index_dates, seasonality=True, seasonality_args = {'trend': 0, 'periods': [2,4,6]}):

    if seasonality:
        seasonality_df = get_seasonality(index_dates, trend=seasonality_args['trend'], periods=seasonality_args['periods'])

    covar_df = pd.concat([seasonality_df], axis=1)

    return covar_df