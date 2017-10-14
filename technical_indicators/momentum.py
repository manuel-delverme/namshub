from technical_indicators import catch_errors
from technical_indicators.function_helper import fill_for_noncomputable_vals


def momentum(data, period):
    """
    Momentum.

    Formula:
    DATA[i] - DATA[i - period]
    """
    catch_errors.check_for_period_error(data, period)

    momentum = [data[idx] - data[idx+1-period] for idx in range(period-1, len(data))]
    momentum = fill_for_noncomputable_vals(data, momentum)
    return momentum
