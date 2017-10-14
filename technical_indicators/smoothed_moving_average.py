import numpy as np
from technical_indicators import catch_errors
from technical_indicators.function_helper import fill_for_noncomputable_vals


def smoothed_moving_average(data, period):
    """
    Smoothed Moving Average.

    Formula:
    smma = avg(data(n)) - avg(data(n)/n) + data(t)/n
    """
    catch_errors.check_for_period_error(data, period)
    smma = [((np.mean(data[idx-(period-1):idx+1]) -
         (np.mean(data[idx-(period-1):idx+1])/period) +
         data[idx])/period) for idx in range(0, len(data))]
    smma = fill_for_noncomputable_vals(data, smma)
    return smma
