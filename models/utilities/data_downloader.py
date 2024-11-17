import os
import re
import numpy as np
import pandas as pd
import shutil
import subprocess

from ast import literal_eval

PARENT_DIR_URL = 'https://raw.githubusercontent.com/Nickolay-Martynenko/GRB-X-Ray-Afterglow/main/dataset/Data/'

SPACES = re.compile(r"\s+")
transform = lambda x: literal_eval("["+SPACES.sub(",", x.strip("[] ")).strip()+"]")

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
    dataset_name:str='features'
    )->tuple:
    """
    Downloads a specified dataset directly from GitHub
    
    Parameters
    ----------
    dataset_name : str, default='features'
        Name of the dataset to be downloaded.
    
    Returns
    -------
    train, val, test : pandas.DataFrames
        Dataframes downloaded from the repository
    """
    
    os.mkdir('./tmp')
    
    train_url = PARENT_DIR_URL+f'train/{dataset_name}.csv'
    subprocess.run(['curl', '-o', './tmp/train.csv',
                    '-s', '--show-error', f'{train_url}'])
    train = dtypes_handler(
        pd.read_csv('./tmp/train.csv', index_col=0).drop('Year', axis=1)
    )

    
    val_url = PARENT_DIR_URL+f'val/{dataset_name}.csv'
    subprocess.run(['curl', '-o', './tmp/val.csv',
                    '-s', '--show-error', f'{val_url}'])
    val = dtypes_handler(
        pd.read_csv('./tmp/val.csv', index_col=0).drop('Year', axis=1)
    )
    
    test_url = PARENT_DIR_URL+f'test/{dataset_name}.csv'
    subprocess.run(['curl', '-o', './tmp/test.csv', '-s',
                    '--show-error', f'{test_url}'])
    test = dtypes_handler(
        pd.read_csv('./tmp/test.csv', index_col=0).drop('Year', axis=1)
    )
    
    shutil.rmtree('./tmp')
    
    print(f'Datasets downloaded\n'+
          f' - train : {len(train)} entries\n'+
          f' - val   : {len(val)} entries\n'+
          f' - test  : {len(test)} entries'
    )
    
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