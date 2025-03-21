import numpy as np
import talib as ta
import logging


class Technicals:
    def __init__(self):
        # Setting up logger
        self.logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG)  # Log level set to DEBUG (you can adjust it)

    def calculate_stoch_rsi_as_tradingview_numpy_fast(self, array, k=3, d=3, period=14):
        """
        Calculate the Stochastic RSI (K and D) using NumPy for efficient computation similar to tradingview as others fails

        Parameters:
        - array (numpy array): Input array of price or close values to compute RSI.
        - k (int, default=3): Period for the %K smoothing (i.e., number of periods for the rolling mean of the Stochastic RSI).
        - d (int, default=3): Period for the %D smoothing (i.e., number of periods for the rolling mean of %K).
        - period (int, default=14): The period for the RSI calculation.

        Returns:
        - tuple: A tuple of two numpy arrays:
            - The smoothed %K values of the Stochastic RSI.
            - The smoothed %D values of the Stochastic RSI.

        Notes:
        - The function calculates the Relative Strength Index (RSI) first, then uses it to compute the Stochastic RSI.
        - The Stochastic RSI is then smoothed using a rolling mean to compute the %K and %D lines.
        - The function uses NumPy's sliding window view for efficient rolling window operations.
        - If the number of values is insufficient for the specified period or smoothing factors (k, d), it returns empty arrays or None.

        Example:
        - Input: array of closing prices.
        - Output: The smoothed %K and %D values of the Stochastic RSI.
        """

        try:
            # Compute RSI
            rsi = ta.RSI(array, period)

            # Remove NaN values from RSI
            rsi = rsi[~np.isnan(rsi)]  # Keeps only valid values

            if len(rsi) >= period:  # Check if enough values for rolling
                # Rolling Min/Max using numpy's sliding_window_view
                rsi_roll = np.lib.stride_tricks.sliding_window_view(rsi, period)
                min_rsi = np.min(rsi_roll, axis=1)
                max_rsi = np.max(rsi_roll, axis=1)

                # Compute Stochastic RSI
                stochastic_rsi = 100 * (rsi[-len(min_rsi):] - min_rsi) / (max_rsi - min_rsi)

                # Smooth K and D
                if len(stochastic_rsi) >= k:
                    stochastic_rsi_K_roll = np.mean(np.lib.stride_tricks.sliding_window_view(stochastic_rsi, k), axis=1)
                    if len(stochastic_rsi_K_roll) >= d:
                        stochastic_rsi_D_roll = np.mean(np.lib.stride_tricks.sliding_window_view(stochastic_rsi_K_roll, d), axis=1)
                    else:
                        stochastic_rsi_D_roll = None  # Not enough values for D
                else:
                    stochastic_rsi_K_roll = None  # Not enough values for K
                    stochastic_rsi_D_roll = None

                return stochastic_rsi_K_roll, stochastic_rsi_D_roll

            else:
                self.logger.error("Not enough RSI values to compute Stochastic RSI")
                return None, None

        except Exception as e:
            self.logger.error(f"An error occurred: {e}")
            return None, None


if __name__ == '__main__':
    # some random data
    array = np.array([84108.71, 84169.82, 84138.83, 84147.58, 84143.41, 84156.24,
                      84098.49, 84000.01, 83992.83, 83978.26, 83990.91, 83968.45,
                      83982.03, 84009.85, 84067.88, 84022.16,
                      84108.71, 84169.82, 84138.83, 84147.58, 84143.41, 84156.24,
                      84098.49, 84000.01, 83992.83, 83978.26, 83990.91, 83968.45,
                      83982.03, 84009.85, 84067.88, 84022.16,
                      84108.71, 84169.82, 84138.83, 84147.58, 84143.41, 84156.24,
                      84098.49, 84000.01, 83992.83, 83978.26, 83990.91, 83968.45,
                      83982.03, 84009.85, 84067.88, 84022.16,
                      84108.71, 84169.82, 84138.83, 84147.58, 84143.41, 84156.24,
                      84098.49, 84000.01, 83992.83, 83978.26, 83990.91, 83968.45,
                      83982.03, 84009.85, 84067.88, 84022.16

                      ])

    technicals = Technicals()
    stochK, stochD = technicals.calculate_stoch_rsi_as_tradingview_numpy_fast(array=array, k=3, d=3, period=14)
