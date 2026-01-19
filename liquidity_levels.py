import pandas as pd
import numpy as np


class LiquidityAnalyzer:
    def __init__(self):
        """Initialize Liquidity Analyzer for BSL/SSL detection."""
        pass

    def find_swing_highs_lows(self, ohlcv_df, swing_length=5):
        """
        Find swing highs and lows.

        Swing High: High that is higher than N bars before and after
        Swing Low: Low that is lower than N bars before and after

        Args:
            ohlcv_df: OHLCV DataFrame
            swing_length: Number of bars to look left/right
        """
        df = ohlcv_df.copy()

        swing_highs = []
        swing_lows = []

        for i in range(swing_length, len(df) - swing_length):
            # Check if current high is a swing high
            current_high = df.iloc[i]['high']
            is_swing_high = True

            for j in range(i - swing_length, i + swing_length + 1):
                if j == i:
                    continue
                if df.iloc[j]['high'] >= current_high:
                    is_swing_high = False
                    break

            if is_swing_high:
                swing_highs.append({
                    'datetime': df.iloc[i]['datetime'],
                    'price': current_high,
                    'index': i,
                    'type': 'swing_high'
                })

            # Check if current low is a swing low
            current_low = df.iloc[i]['low']
            is_swing_low = True

            for j in range(i - swing_length, i + swing_length + 1):
                if j == i:
                    continue
                if df.iloc[j]['low'] <= current_low:
                    is_swing_low = False
                    break

            if is_swing_low:
                swing_lows.append({
                    'datetime': df.iloc[i]['datetime'],
                    'price': current_low,
                    'index': i,
                    'type': 'swing_low'
                })

        return {
            'swing_highs': pd.DataFrame(swing_highs),
            'swing_lows': pd.DataFrame(swing_lows)
        }

    def identify_liquidity_levels(self, ohlcv_df, swing_length=5):
        """
        Identify Buy-Side Liquidity (BSL) and Sell-Side Liquidity (SSL) levels.

        BSL = Above swing highs (where buy stop orders likely sit)
        SSL = Below swing lows (where sell stop orders likely sit)

        Args:
            ohlcv_df: OHLCV DataFrame
            swing_length: Swing detection parameter
        """
        swings = self.find_swing_highs_lows(ohlcv_df, swing_length)

        bsl_levels = []
        ssl_levels = []

        # BSL = Swing highs (liquidity resting above)
        if len(swings['swing_highs']) > 0:
            for idx, swing in swings['swing_highs'].iterrows():
                bsl_levels.append({
                    'datetime': swing['datetime'],
                    'price': swing['price'],
                    'type': 'BSL',
                    'description': 'Buy-Side Liquidity (stops above high)'
                })

        # SSL = Swing lows (liquidity resting below)
        if len(swings['swing_lows']) > 0:
            for idx, swing in swings['swing_lows'].iterrows():
                ssl_levels.append({
                    'datetime': swing['datetime'],
                    'price': swing['price'],
                    'type': 'SSL',
                    'description': 'Sell-Side Liquidity (stops below low)'
                })

        return {
            'bsl': pd.DataFrame(bsl_levels),
            'ssl': pd.DataFrame(ssl_levels),
            'all_levels': pd.DataFrame(bsl_levels + ssl_levels)
        }

    def detect_liquidity_sweeps(self, ohlcv_df, liquidity_levels):
        """
        Detect when price sweeps liquidity levels.

        Sweep occurs when:
        - Price briefly moves above BSL then reverses (liquidity grab)
        - Price briefly moves below SSL then reverses

        Args:
            ohlcv_df: OHLCV DataFrame
            liquidity_levels: Output from identify_liquidity_levels()
        """
        sweeps = []

        bsl_df = liquidity_levels['bsl']
        ssl_df = liquidity_levels['ssl']

        for idx, candle in ohlcv_df.iterrows():
            # Check BSL sweeps (price went above then closed below)
            for _, bsl in bsl_df.iterrows():
                if bsl['datetime'] < candle['datetime']:
                    # BSL sweep: high touches BSL but close is below
                    if candle['high'] >= bsl['price'] and candle['close'] < bsl['price']:
                        sweeps.append({
                            'datetime': candle['datetime'],
                            'liquidity_price': bsl['price'],
                            'candle_high': candle['high'],
                            'candle_close': candle['close'],
                            'type': 'BSL_sweep',
                            'description': 'Liquidity sweep above BSL (bearish)'
                        })

            # Check SSL sweeps (price went below then closed above)
            for _, ssl in ssl_df.iterrows():
                if ssl['datetime'] < candle['datetime']:
                    # SSL sweep: low touches SSL but close is above
                    if candle['low'] <= ssl['price'] and candle['close'] > ssl['price']:
                        sweeps.append({
                            'datetime': candle['datetime'],
                            'liquidity_price': ssl['price'],
                            'candle_low': candle['low'],
                            'candle_close': candle['close'],
                            'type': 'SSL_sweep',
                            'description': 'Liquidity sweep below SSL (bullish)'
                        })

        return pd.DataFrame(sweeps)

    def find_fair_value_gaps(self, ohlcv_df):
        """
        Find Fair Value Gaps (FVG / Imbalance).

        Bullish FVG: Gap between candle[i-2].high and candle[i].low
        Bearish FVG: Gap between candle[i-2].low and candle[i].high

        Args:
            ohlcv_df: OHLCV DataFrame
        """
        fvgs = []

        for i in range(2, len(ohlcv_df)):
            candle_prev2 = ohlcv_df.iloc[i - 2]
            candle_prev1 = ohlcv_df.iloc[i - 1]
            candle_current = ohlcv_df.iloc[i]

            # Bullish FVG: current low > prev2 high (gap up)
            if candle_current['low'] > candle_prev2['high']:
                fvgs.append({
                    'datetime': candle_current['datetime'],
                    'gap_top': candle_current['low'],
                    'gap_bottom': candle_prev2['high'],
                    'gap_size': candle_current['low'] - candle_prev2['high'],
                    'type': 'bullish_fvg',
                    'description': 'Bullish Fair Value Gap'
                })

            # Bearish FVG: current high < prev2 low (gap down)
            if candle_current['high'] < candle_prev2['low']:
                fvgs.append({
                    'datetime': candle_current['datetime'],
                    'gap_top': candle_prev2['low'],
                    'gap_bottom': candle_current['high'],
                    'gap_size': candle_prev2['low'] - candle_current['high'],
                    'type': 'bearish_fvg',
                    'description': 'Bearish Fair Value Gap'
                })

        return pd.DataFrame(fvgs)

    def calculate_support_resistance(self, ohlcv_df, num_levels=5):
        """
        Calculate key support and resistance levels based on price clusters.

        Args:
            ohlcv_df: OHLCV DataFrame
            num_levels: Number of S/R levels to identify
        """
        # Collect all highs and lows
        all_highs = ohlcv_df['high'].values
        all_lows = ohlcv_df['low'].values
        all_closes = ohlcv_df['close'].values

        price_points = np.concatenate([all_highs, all_lows, all_closes])

        # Use KDE or clustering to find price clusters
        # Simple approach: histogram-based clustering
        hist, bin_edges = np.histogram(price_points, bins=50)

        # Find peaks in histogram (high-frequency price zones)
        peak_indices = []
        for i in range(1, len(hist) - 1):
            if hist[i] > hist[i - 1] and hist[i] > hist[i + 1]:
                peak_indices.append(i)

        # Sort by frequency and take top N
        peak_indices_sorted = sorted(peak_indices, key=lambda x: hist[x], reverse=True)[:num_levels]

        levels = []
        for idx in peak_indices_sorted:
            price = (bin_edges[idx] + bin_edges[idx + 1]) / 2
            levels.append({
                'price': price,
                'frequency': hist[idx],
                'type': 'support_resistance'
            })

        return pd.DataFrame(levels).sort_values('price', ascending=False)
