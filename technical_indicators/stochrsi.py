import numpy as np
from technical_indicators.function_helper import fill_for_noncomputable_vals
from technical_indicators.relative_strength_index import relative_strength_index


def stochrsi(data, period):
    """
    StochRSI.

    Formula:
    SRSI = ((RSIt - RSI LOW) / (RSI HIGH - LOW RSI)) * 100
    """
    rsi = relative_strength_index(data, period)[period:]
    stochrsi = [100 * ((rsi[idx] - np.min(rsi[idx+1-period:idx+1])) / (np.max(rsi[idx+1-period:idx+1]) - np.min(rsi[idx+1-period:idx+1]))) for idx in range(period-1, len(rsi))]
    stochrsi = fill_for_noncomputable_vals(data, stochrsi)
    return stochrsi
