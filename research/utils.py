
import pandas as pd
import numpy as np

from IPython.display import display, HTML
display(HTML(" <style>.container{width:95% !important;}</style> "))
from IPython.display import display_html
from itertools import chain, cycle


def display_side_by_side(*args, titles = cycle([''])):
    html_str = ''
    for df, title in zip(args, chain(titles, cycle(['</br>']))):
        html_str += '<th style="text-align:center"> <td style = "vertical-align:top"> '
        html_str += f'<h4 style = "text-align: center;"> {title} </h4>'
        html_str += df.to_html().replace('table', 'table style="display:inline"')
        html_str += '</td></th>'
    display_html(html_str, raw=True)

def curate_single_future_data(raw_data_filename, sheet_name="CL"):

    df = pd.read_excel(raw_data_filename, sheet_name=sheet_name)
    if sheet_name in ['CL',
                      'CO',
                      'HO',
                      'XB',
                      'QS']:
        winds = [1 ,3 ,6]
    else:
        winds = ['1_3' ,'3_6' ,'6_12']

    px_cols = [sheet_name +str(i) for i in winds ]
    df1 = df[['Date', px_cols[0]]]
    df1.rename(columns = { px_cols[0]: 'Price'}, inplace = True)
    df1['SecurityId'] = px_cols[0]
    df3 = df[['Date1', px_cols[1]]]
    df3.rename(columns = {'Date1': 'Date',
                          px_cols[1]: 'Price'}, inplace = True)
    df3['SecurityId'] = px_cols[1]
    df6 = df[['Date2', px_cols[2]]]
    df6.rename(columns = {'Date2': 'Date',
                          px_cols[2]: 'Price'}, inplace = True)
    df6['SecurityId'] = px_cols[2]


    df1.dropna(inplace = True)
    df3.dropna(inplace = True)
    df6.dropna(inplace = True)

    data =  pd.concat([df1, df3, df6]).reset_index(drop = True)
    return data


def get_correlation(data, ycol, xcols, method='pearson'):
    corr_dict = {}
    for xc in xcols:
        corr_dict[xc] = [data[[ycol, xc]].corr(method=method).iloc[0, 1]]

    corr_df = pd.DataFrame.from_dict(corr_dict).T.reset_index()
    ncol = 'Correlation'
    corr_df.columns = ['ETF', ncol]
    corr_df['LatentFactorName'] = ycol
    corr_df.sort_values(by=ncol, ascending=False, inplace=True)

    return corr_df