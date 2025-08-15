import pandas as pd
from itertools import product
from typing import List
from datetime import date

def get_cartesian_product(securities: list[str],
                         dates: list[date]) -> pd.DataFrame:
    """
    Compute the Cartesian product between a list of security names and a list of dates,
    returning the result as a DataFrame with columns 'tradeDate' and 'Name'.

    Args:
        securities (List[str]): List of security names.
        dates (List[date]): List of dates.

    Returns:
        pd.DataFrame: DataFrame containing the Cartesian product with columns 'tradeDate' and 'Name'.
    """
    cartesian_product = list(product(dates, securities))
    return pd.DataFrame(cartesian_product, columns=['tradeDate', 'Name'])