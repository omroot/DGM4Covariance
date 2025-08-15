import datetime
import numpy as np
import pandas as pd
from pandas_market_calendars import get_calendar

def get_timeOfDay_as_float(dt: datetime.datetime) -> float:
    """ Transform a datetime into a float """
    return dt.hour +dt.minute / 60 + dt.second / (60*60)

def get_first_of_next_month(anydate: datetime.date)->datetime.date:
    """ Returns the first day of the next month relative to the given day. """
    if anydate.month !=12:
        return datetime.date(anydate.year, anydate.month+1,1)
    return datetime.date(anydate.year +1,1,1)
def get_last_day_of_month(any_day: datetime.date)->datetime.date:
    """" Returns the last day of the month relative to the given day. """
    next_month = any_day.replace(day=28)+datetime.timedelta(days=4)
    return next_month




def get_holidays(
    exchange_name: str,
    start_date: datetime.date,
    end_date:  datetime.date
) -> list[ datetime.date]:
    """
    Get holidays for a specific exchange between start_date and end_date.

    Parameters:
        exchange_name (str): Name of the exchange (e.g., 'XNYS' for NYSE).
        start_date ( datetime.date): Start date for holiday retrieval.
        end_date ( datetime.date): End date for holiday retrieval.

    Returns:
        List[ datetime.date]: List of holidays between start_date and end_date.
    """

    holidays = pd.to_datetime(pd.Series(get_calendar('NYSE').holidays().holidays)).dt.date

    return holidays[( holidays>= start_date ) & (holidays <= end_date ) ].unique().tolist()

def get_business_dates(start_date: datetime.date,
                            end_date:  datetime.date
                        ) -> list[ datetime.date]:
    """ Get  business dates between two dates """
    dates = pd.date_range(start_date, end_date, freq = 'D')
    weekend_mask = (dates.dayofweek ==5) | (dates.dayofweek == 6)
    holidays = get_holidays(exchange_name =  'NYSE', start_date = start_date, end_date = end_date)
    holiday_mask = dates.isin(holidays)
    non_business_day_mask = weekend_mask | holiday_mask
    business_dates = pd.to_datetime(pd.Series ( list( dates[~non_business_day_mask] ) ) ).dt.date.tolist()

    return business_dates

