#!/usr/bin/env python3
"""
Risk Metrics Calculator for Crypto Assets
Calculates risk scores, volatility, volume analysis, and trend for multiple coins
"""
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json


class RiskMetricsCalculator:
    def __init__(self):
        self.exchange = self._initialize_exchange()

    def _initialize_exchange(self):
        """Initialize exchange with fallback options"""
        exchanges_to_try = [
            ('binance', ccxt.binance),
            ('kraken', ccxt.kraken),
            ('bybit', ccxt.bybit),
        ]

        for name, exchange_class in exchanges_to_try:
            try:
                exchange = exchange_class({
                    'enableRateLimit': True,
                    'options': {'defaultType': 'spot'}
                })
                # Test if exchange works
                exchange.load_markets()
                print(f"✓ RiskMetrics using {name} exchange")
                return exchange
            except Exception as e:
                print(f"⚠ {name} failed: {str(e)[:100]}")
                continue

        # Fallback to binance
        print("⚠ All exchanges failed, using binance as fallback")
        return ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })

    def fetch_ohlcv(self, symbol, timeframe='15m', days=10):
        """Fetch OHLCV data for a symbol"""
        try:
            since = self.exchange.parse8601((datetime.now() - timedelta(days=days)).isoformat())
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)

            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            return None

    def calculate_volatility(self, df, window=50):
        """Calculate volatility percentage"""
        if df is None or len(df) < window:
            return 0

        returns = df['close'].pct_change()
        volatility = returns.std() * np.sqrt(96)  # Annualized for 15m (96 periods per day)
        return volatility * 100  # Convert to percentage

    def calculate_volume_strength(self, df, window=50):
        """Calculate volume strength relative to average"""
        if df is None or len(df) < window:
            return 1.0

        avg_volume = df['volume'].tail(window).mean()
        recent_volume = df['volume'].tail(10).mean()

        if avg_volume == 0:
            return 1.0

        return recent_volume / avg_volume

    def calculate_trend(self, df, short_window=20, long_window=50):
        """Calculate trend direction using EMAs"""
        if df is None or len(df) < long_window:
            return 0

        ema_short = df['close'].ewm(span=short_window, adjust=False).mean()
        ema_long = df['close'].ewm(span=long_window, adjust=False).mean()

        current_short = ema_short.iloc[-1]
        current_long = ema_long.iloc[-1]

        trend_strength = ((current_short - current_long) / current_long) * 100
        return trend_strength

    def calculate_rsi(self, df, period=14):
        """Calculate RSI"""
        if df is None or len(df) < period:
            return 50

        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]

    def calculate_risk_score(self, volatility, volume_strength, rsi, trend_strength):
        """
        Calculate overall risk score (0-100)
        Lower score = lower risk
        Higher score = higher risk
        """
        # Volatility component (0-40 points)
        vol_score = min(volatility * 2, 40)  # High volatility = high risk

        # Volume component (0-20 points)
        # Low volume = higher risk (liquidity risk)
        if volume_strength < 0.5:
            volume_score = 20
        elif volume_strength < 1.0:
            volume_score = 10
        else:
            volume_score = 0

        # RSI component (0-20 points)
        # Extreme RSI values = higher risk
        if rsi > 70 or rsi < 30:
            rsi_score = 20
        elif rsi > 60 or rsi < 40:
            rsi_score = 10
        else:
            rsi_score = 0

        # Trend uncertainty (0-20 points)
        # Weak or no trend = higher risk
        trend_score = max(0, 20 - abs(trend_strength) * 5)

        total_risk = vol_score + volume_score + rsi_score + trend_score
        return min(int(total_risk), 100)

    def get_volatility_label(self, volatility):
        """Convert volatility percentage to label"""
        if volatility < 20:
            return "Low"
        elif volatility < 40:
            return "Medium"
        else:
            return "High"

    def get_volume_label(self, volume_strength):
        """Convert volume strength to label"""
        if volume_strength < 0.5:
            return "Low"
        elif volume_strength < 1.0:
            return "Medium"
        elif volume_strength < 1.5:
            return "High"
        else:
            return "Very High"

    def get_trend_label(self, trend_strength):
        """Convert trend strength to label"""
        if trend_strength > 2:
            return "Bullish"
        elif trend_strength < -2:
            return "Bearish"
        else:
            return "Neutral"

    def calculate_metrics_for_coin(self, symbol):
        """Calculate all metrics for a single coin"""
        print(f"📊 Calculating metrics for {symbol}...")

        df = self.fetch_ohlcv(symbol, timeframe='15m', days=10)

        if df is None:
            return None

        volatility = self.calculate_volatility(df)
        volume_strength = self.calculate_volume_strength(df)
        trend_strength = self.calculate_trend(df)
        rsi = self.calculate_rsi(df)

        risk_score = self.calculate_risk_score(volatility, volume_strength, rsi, trend_strength)

        metrics = {
            'risk': risk_score,
            'volatility': self.get_volatility_label(volatility),
            'volume': self.get_volume_label(volume_strength),
            'trend': self.get_trend_label(trend_strength),
            'raw_data': {
                'volatility_pct': round(volatility, 2),
                'volume_strength': round(volume_strength, 2),
                'trend_strength': round(trend_strength, 2),
                'rsi': round(rsi, 2)
            }
        }

        print(f"  ✓ Risk: {risk_score} | Vol: {metrics['volatility']} | "
              f"Volume: {metrics['volume']} | Trend: {metrics['trend']}")

        return metrics

    def calculate_all_metrics(self, coins):
        """Calculate metrics for multiple coins"""
        results = {}

        for coin_name, symbol in coins.items():
            metrics = self.calculate_metrics_for_coin(symbol)
            if metrics:
                results[coin_name] = metrics
            else:
                # Fallback data if fetch fails
                results[coin_name] = {
                    'risk': 50,
                    'volatility': 'Medium',
                    'volume': 'Medium',
                    'trend': 'Neutral'
                }

        return results

    def save_to_json(self, metrics, output_file):
        """Save metrics to JSON file"""
        # Remove raw_data for the web output
        clean_metrics = {}
        for coin, data in metrics.items():
            clean_metrics[coin] = {
                'risk': data['risk'],
                'volatility': data['volatility'],
                'volume': data['volume'],
                'trend': data['trend']
            }

        with open(output_file, 'w') as f:
            json.dump(clean_metrics, f, indent=2)

        print(f"✓ Saved metrics to {output_file}")


def main():
    """Test the calculator"""
    calculator = RiskMetricsCalculator()

    coins = {
        'SOL': 'SOL/USDT',
        'ETH': 'ETH/USDT',
        'BNB': 'BNB/USDT',
        'DASH': 'DASH/USDT'
    }

    print("\n" + "="*70)
    print("  RISK METRICS CALCULATOR")
    print("="*70 + "\n")

    metrics = calculator.calculate_all_metrics(coins)

    output_file = "/Users/muhamedalanc/crypto-website/risk_metrics.json"
    calculator.save_to_json(metrics, output_file)

    print("\n✓ Risk metrics calculation complete!")


if __name__ == "__main__":
    main()
