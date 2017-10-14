from technical_indicators.standard_deviation import standard_deviation as sd
from technical_indicators.standard_variance import standard_variance as sv


def volatility(data, period):
    """
    Volatility.

    Formula:
    SDt / SVt
    """
    volatility = sd(data, period) / sv(data, period)
    return volatility
