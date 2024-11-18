import os
import re
import numpy as np
import pandas as pd
import shutil
import subprocess

from ast import literal_eval

PARENT_DIR_URL = (
    'https://raw.githubusercontent.com/'+
    'Nickolay-Martynenko/GRB-X-Ray-Afterglow/'+
    'main/dataset/'
)

SPACES = re.compile(r"\s+")
transform = lambda x: literal_eval(
    "["+SPACES.sub(",", x.strip("[] ")).strip()+"]"
)

def dtypes_handler(
        df:pd.DataFrame
    )->pd.DataFrame:
    """
    Handles list occurences in the dataframe.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to be processed.
        
    Returns
    -------
    df : pd.DataFrame
        Processed dataframe.
    """
    
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col]=df[col].apply(transform)
            
    return df

def train_val_test_downloader(
    dataset_name:str='features',
    download_labels:bool=True
    )->tuple:
    """
    Downloads a specified dataset directly from GitHub
    
    Parameters
    ----------
    dataset_name : str, default='features'
        Name of the dataset to be downloaded.
    load_labels : bool, default=True
        If True, Swift analysis data is also downloaded
        and returned as a separate dataframe.
    Returns
    -------
    train, val, test : pandas.DataFrames
        Dataframes downloaded from the repository
    labels : pandas.DataFrame, optional
        Only returned if `load_labels`=True.
        Swift analysis data.
    """
    shutil.rmtree('./tmp', ignore_errors=True)
    os.mkdir('./tmp')
    
    train_url = PARENT_DIR_URL+f'Data/train/{dataset_name}.csv'
    subprocess.run(['curl', '-o', './tmp/train.csv',
                    '-s', '--show-error', f'{train_url}'])
    train = dtypes_handler(
        pd.read_csv('./tmp/train.csv', index_col=0).drop('Year', axis=1)
    )

    
    val_url = PARENT_DIR_URL+f'Data/val/{dataset_name}.csv'
    subprocess.run(['curl', '-o', './tmp/val.csv',
                    '-s', '--show-error', f'{val_url}'])
    val = dtypes_handler(
        pd.read_csv('./tmp/val.csv', index_col=0).drop('Year', axis=1)
    )
    
    test_url = PARENT_DIR_URL+f'Data/test/{dataset_name}.csv'
    subprocess.run(['curl', '-o', './tmp/test.csv', '-s',
                    '--show-error', f'{test_url}'])
    test = dtypes_handler(
        pd.read_csv('./tmp/test.csv', index_col=0).drop('Year', axis=1)
    )

    if download_labels:
        labels_url = PARENT_DIR_URL+'Data/GRBtable.csv'
        subprocess.run(['curl', '-o', './tmp/labels.csv', '-s',
                        '--show-error', f'{labels_url}'])
        labels = pd.read_csv('./tmp/labels.csv',
            index_col=0, delimiter=';', header=0)
        labels.replace({'N/A ': pd.NaT}, inplace=True)
        labels['Flares'] = labels['Flares'].apply(literal_eval)
        labels['FlaresFlag'] = labels['Flares'].astype(bool).astype(int)
        index = np.hstack((train.index, val.index, test.index))
        labels = labels.loc[index, :]
    
    shutil.rmtree('./tmp')
    
    print(f'Datasets downloaded\n'+
          f' - train  : {len(train)} entries\n'+
          f' - val    : {len(val)} entries\n'+
          f' - test   : {len(test)} entries'+
          f'\n - labels : {len(labels)} entries' if download_labels else ''
    )
    
    if download_labels:
        return (train, val, test, labels)
    else:
        return (train, val, test)

def choose_one_column(df:pd.DataFrame, column:str)->pd.DataFrame:
    """
    Transforms a dataframe with a column of lists
    to a standard table dataframe, ignoring all the 
    other columns and without assigning any names 
    to the extracted features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to be processed.
    column : str
        Column to be converted to a table.
        
    Returns
    -------
    df_copy : pd.DataFrame
        Processed dataframe.
    """
    
    data = np.array(df[column].values.tolist())
    index = df.index
    df_copy = pd.DataFrame(data=data, index=index)
    
    return df_copy

def download_single_LC(
    event_name:str,
    modes:list=['PC_incbad', 'WT_incbad'],
    )->tuple:
    """
    Downloads lightcurve for a single event
    directly from the repository dataset.
    Output is designed to follow the expected
    matplotlib.pyplot.errorbar input pattern

    Parameters
    ----------
    event_name : str
        The name of event to be downloaded
    modes : list
        The available modes of data collection

    Returns
    -------
    x : np.ndarray
        Timestamps
    y : np.ndarray
        Source count rate 
    yerr : np.ndarray
        -/+ source count rate errors.
    xerr : np.ndarray
        -/+ timestamps errors.
    """
    shutil.rmtree('./tmp', ignore_errors=True)
    os.mkdir('./tmp')

    dataframes = []

    for mode in modes:
        url = PARENT_DIR_URL+f'SwiftXRT/{mode}/{event_name}.json'
        subprocess.run([
            'curl', '-o',
            f'./tmp/{mode}_{event_name}.json',
            '-s', '--show-error', f'{url}']
        )
        df = pd.read_json(f'./tmp/{mode}_{event_name}.json')
        if len(df) > 0:
            dataframes.append(df.loc[:,
                ['Time', 'TimeNeg', 'TimePos',
                 'Rate', 'RatePos', 'RateNeg']
                ]
            )
    dataframe = pd.concat(
        dataframes, axis=0, ignore_index=True
    ).sort_values(by='Time')

    x = dataframe['Time'].values
    xerr = dataframe[['TimeNeg', 'TimePos']].values
    xerr[0, :] *= -1

    y = dataframe['Rate'].values
    yerr = dataframe[['RateNeg', 'RatePos']].values
    yerr[0, :] *= -1

    return x, y, yerr, xerr





