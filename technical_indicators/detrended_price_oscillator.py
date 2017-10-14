import numpy as np
from technical_indicators import catch_errors
from technical_indicators.function_helper import fill_for_noncomputable_vals


def detrended_price_oscillator(data, period):
    """
    Detrended Price Oscillator.

    Formula:
    DPO = DATA[i] - Avg(DATA[period/2 + 1])
    """
    catch_errors.check_for_period_error(data, period)
    period = int(period)
    dop = [data[idx] - np.mean(data[idx+1-((period/2)+1):idx+1]) for idx in range(period-1, len(data))]
    dop = fill_for_noncomputable_vals(data, dop)
    return dop
