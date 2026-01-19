#!/usr/bin/env python3
"""
GitHub Actions Update Script
Generates all dashboards and updates files
"""
import sys
from datetime import datetime
import pytz
from altcoin_ratio import AltcoinRatioCalculator
from altcoin_visualizer import AltcoinRatioVisualizer
from liquidity_levels import LiquidityAnalyzer
from risk_metrics_calculator import RiskMetricsCalculator
from detailed_risk_analyzer import DetailedRiskAnalyzer

def main():
    print("="*70)
    print("  GITHUB ACTIONS - DASHBOARD UPDATE")
    print("="*70)

    # Initialize
    calculator = AltcoinRatioCalculator()
    visualizer = AltcoinRatioVisualizer()
    liquidity = LiquidityAnalyzer()
    risk_calculator = RiskMetricsCalculator()
    detailed_analyzer = DetailedRiskAnalyzer()

    overlay_symbol = 'ETH/USDT'
    overlay_name = 'ETH'
    timeframe = "15m"
    days = 10
    orderbook_candles = 100

    dubai_tz = pytz.timezone('Asia/Dubai')
    dubai_time = datetime.now(dubai_tz).strftime('%H:%M:%S')

    print(f"\n⏰ Dubai Time: {dubai_time}")
    print(f"📊 Fetching data...\n")

    # Fetch market data with retry
    market_data = None
    for retry in range(3):
        market_data = calculator.fetch_market_caps()
        if market_data:
            break
        print(f"⚠ Market data fetch failed, retry {retry+1}/3...")

    if not market_data:
        print("✗ Market data failed after 3 retries")
        sys.exit(1)

    # Fetch overlay price data with retry
    overlay_df = None
    for retry in range(3):
        overlay_df = calculator.fetch_overlay_price(overlay_symbol, timeframe, days=days)
        if overlay_df is not None:
            break
        print(f"⚠ Candle data fetch failed, retry {retry+1}/3...")

    if overlay_df is None:
        print("✗ Candle data failed after 3 retries")
        sys.exit(1)

    print(f"✅ FRESH DATA: {len(overlay_df)} candles (latest: {overlay_df['datetime'].iloc[-1]})")

    # Calculate ratio
    ratio_df = calculator.calculate_synthetic_altcoin_ratio(overlay_df, market_data, overlay_name)
    ratio_df = calculator.calculate_indicators(ratio_df)

    # Calculate S/R
    sr_result = calculator.find_auto_sr_levels_pinescript(
        ratio_df, left_right=3, tolerance=30.0,
        max_supports=4, max_resistances=4
    )

    # BSL/SSL
    ratio_ohlc = ratio_df[['datetime', 'ar_open', 'ar_high', 'ar_low', 'ar_close']].copy()
    ratio_ohlc.columns = ['datetime', 'open', 'high', 'low', 'close']
    swings = liquidity.find_swing_highs_lows(ratio_ohlc, swing_length=10)

    bsl = swings['swing_highs']['price'].iloc[-1] if len(swings['swing_highs']) > 0 else None
    ssl = swings['swing_lows']['price'].iloc[-1] if len(swings['swing_lows']) > 0 else None

    # Footprint
    footprint_df = calculator.calculate_footprint(ratio_df, lookback=50)

    # Create main dashboard (without orderbook - not available in GitHub Actions)
    print(f"📝 Generating main dashboard...")
    visualizer.create_combined_chart(
        btc_df=overlay_df,
        ratio_df=ratio_df,
        orderbook_data=None,  # No live orderbook in GitHub Actions
        support_levels=sr_result['supports'],
        resistance_levels=sr_result['resistances'],
        bsl_ssl={'bsl': bsl, 'ssl': ssl},
        footprint_df=footprint_df,
        orderbook_candles=orderbook_candles,
        output_file="altcoin_combined_eth_live.html"
    )
    print("✓ Main dashboard created")

    # Calculate Risk Metrics
    print("📊 Calculating risk metrics...")
    coins = {
        'SOL': 'SOL/USDT',
        'ETH': 'ETH/USDT',
        'BNB': 'BNB/USDT',
        'DASH': 'DASH/USDT'
    }
    risk_metrics = risk_calculator.calculate_all_metrics(coins)
    risk_calculator.save_to_json(risk_metrics, "risk_metrics.json")

    # Generate Detailed Risk Analyzer Dashboards (every run in GitHub Actions)
    print("📊 Generating detailed analyzer dashboards...")
    for coin_name, symbol in coins.items():
        analyzer_file = f"{coin_name.lower()}_risk_analyzer.html"
        try:
            detailed_analyzer.create_dashboard(symbol, coin_name, analyzer_file)
        except Exception as e:
            print(f"⚠ Analyzer error for {coin_name}: {e}")

    dubai_time_now = datetime.now(dubai_tz).strftime('%H:%M:%S')
    print(f"\n✅ All dashboards updated successfully!")
    print(f"✅ Dubai Time: {dubai_time_now}")
    print("="*70)

if __name__ == "__main__":
    main()
