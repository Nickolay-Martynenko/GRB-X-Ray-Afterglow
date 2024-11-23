import numpy as np
import pandas as pd 
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression

def explorer(df:pd.DataFrame, target_col:str='numbreaks'):
    """
    Explores correlations between the extracted features
    and target column in the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe. Must contain extracted features in 
        columns named 'feature_0', 'feature_1' and so on.
        Also must contain 'sample' column, which defines
        train-val-test split, and `target_col` column.

    target_col : str, default='numbreaks'
        The target variable to be predicted from extracted features.

    Returns
    -------
    scaler : RobustScaler
        Fitted RobustScaler object/

    LinReg : LinearRegression
        Fitted LinearRegression object.
    """

    features = [col for col in df.columns if col.startswith('feature_')]
    fragment = df[
        features+
        ['numbreaks', 'sample']
    ]
    
    print('correlation between latent features and '+
      f'numbreaks: {np.corrcoef(fragment.drop('sample', axis=1).values, rowvar=False)[:-1, -1]}'
    )
    
    X_train, y_train, X_val, y_val, X_test, y_test = (
        fragment.loc[fragment['sample'=='train'], features],
        fragment.loc[fragment['sample'=='train'], 'numbreaks'],
        fragment.loc[fragment['sample'=='val'], features],
        fragment.loc[fragment['sample'=='val'], 'numbreaks'],
        fragment.loc[fragment['sample'=='test'], features],
        fragment.loc[fragment['sample'=='test'], 'numbreaks'],
    )
    
    scaler = RobustScaler().fit(X_train)
    X_train, X_val, X_test = tuple(
        map(
            scaler.transform, (X_train, X_val, X_test)
        )
    )
    LinReg = LinearRegression().fit(X_train, y_train)
    
    print('Linear Regression CoD score'+
          '\nfor numbreaks prediction problem'+
          '\nusing latent features as X:'+
          f'\n    train : {LinReg.score(X_train, y_train)}'+
          f'\n    val   : {LinReg.score(X_val, y_val)}'+
          f'\n    test  : {LinReg.score(X_test, y_test)}'
    )

    return scaler, LinReg