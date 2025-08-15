from typing import Optional
import pandas as pd
import numpy as np
import scipy.stats as ss
from sklearn.metrics import mutual_info_score
def bin_summary_of_xy(
        x:pd.Series,
        y:pd.Series,
        k: int,
        unique_flag: Optional[bool]=True

) ->pd.DataFrame:


    if unique_flag:
        _, x_bounds = pd.qcut(
            np.unique(x), k, labels = False, retbins=True

        )
        x_binned = pd.cut(x, bins=x_bounds, right=False)
    else:
        x_binned, x_bounds = pd.qcut(x, k, labels= False, retbins=True)

    df = pd.DataFrame({ "x_binned": x_binned, "x": x, "y": y })
    aggregating_functions = ["mean", "std", "count"]
    bin_analytics = (

        df.groupby("x_binned").agg(aggregating_functions).reset_index()
    )
    bin_analytics.columns = [
        "x_binned",
        "x_mean",
        "x_std",
        "x_count",
        "y_mean",
        "y_std",
        "y_count"
    ]
    bin_analytics["x_se"] = bin_analytics["x_std"]/np.sqrt(bin_analytics["x_count"])
    bin_analytics["y_se"] = bin_analytics["y_std"] / np.sqrt(bin_analytics["y_count"])

    bin_analytics["lower_bound"] =  list(x_bounds)[:-1]
    bin_analytics["upper_bound"] =  list(x_bounds)[1:]

    return bin_analytics

