import numpy as np
import warnings
from technical_indicators.function_helper import fill_for_noncomputable_vals
from technical_indicators import catch_errors


def simple_moving_average(data, period):
    """
    Simple Moving Average.

    Formula:
    SUM(data / N)
    """
    catch_errors.check_for_period_error(data, period)
    # Mean of Empty Slice RuntimeWarning doesn't affect output so it is
    # supressed
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        sma = [np.mean(data[idx-(period-1):idx+1]) for idx in range(0, len(data))]
    sma = fill_for_noncomputable_vals(data, sma)
    return sma
