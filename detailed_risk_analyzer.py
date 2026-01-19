#!/usr/bin/env python3
"""
Detailed Risk Analyzer - Creates comprehensive Plotly dashboards for each coin
Similar to CHRP Risk Analyzer but for SOL, ETH, BNB, DASH
"""
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json


class DetailedRiskAnalyzer:
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
                print(f"✓ DetailedRiskAnalyzer using {name} exchange")
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

    def fetch_ohlcv(self, symbol, timeframe='1h', days=30):
        """Fetch OHLCV data"""
        try:
            since = self.exchange.parse8601((datetime.now() - timedelta(days=days)).isoformat())
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)

            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            print(f"❌ Error fetching {symbol}: {e}")
            return None

    def calculate_indicators(self, df):
        """Calculate technical indicators"""
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['histogram'] = df['macd'] - df['signal']

        # Bollinger Bands
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['std_20'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['sma_20'] + (df['std_20'] * 2)
        df['bb_lower'] = df['sma_20'] - (df['std_20'] * 2)

        # EMAs
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()

        # Volatility
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(24) * 100

        # Volume MA
        df['volume_ma'] = df['volume'].rolling(window=20).mean()

        return df

    def calculate_risk_factors(self, df):
        """Calculate detailed risk factors"""
        current_price = df['close'].iloc[-1]

        # Price metrics
        high_52w = df['high'].tail(365*24).max() if len(df) > 365*24 else df['high'].max()
        low_52w = df['low'].tail(365*24).min() if len(df) > 365*24 else df['low'].min()
        price_from_high = ((current_price - high_52w) / high_52w) * 100
        price_from_low = ((current_price - low_52w) / low_52w) * 100

        # Volatility
        current_volatility = df['volatility'].iloc[-1]
        avg_volatility = df['volatility'].tail(720).mean()  # 30 days

        # RSI
        current_rsi = df['rsi'].iloc[-1]

        # MACD
        macd_histogram = df['histogram'].iloc[-1]

        # Volume
        current_volume = df['volume'].iloc[-1]
        avg_volume = df['volume_ma'].iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1

        # Trend (EMA crossovers)
        ema_20 = df['ema_20'].iloc[-1]
        ema_50 = df['ema_50'].iloc[-1]
        ema_200 = df['ema_200'].iloc[-1]

        trend_score = 0
        if ema_20 > ema_50 > ema_200:
            trend = "Strong Bullish"
            trend_score = 100
        elif ema_20 > ema_50:
            trend = "Bullish"
            trend_score = 75
        elif ema_20 < ema_50 < ema_200:
            trend = "Strong Bearish"
            trend_score = 0
        elif ema_20 < ema_50:
            trend = "Bearish"
            trend_score = 25
        else:
            trend = "Neutral"
            trend_score = 50

        # Overall risk score (0-100, lower = lower risk)
        risk_factors = {
            'volatility_risk': min(current_volatility * 2, 40),  # 0-40
            'rsi_risk': 20 if (current_rsi > 70 or current_rsi < 30) else (10 if (current_rsi > 60 or current_rsi < 40) else 0),  # 0-20
            'volume_risk': 0 if volume_ratio > 1 else (10 if volume_ratio > 0.5 else 20),  # 0-20
            'trend_risk': max(0, 20 - (trend_score / 5))  # 0-20
        }

        overall_risk = sum(risk_factors.values())

        return {
            'current_price': current_price,
            'high_52w': high_52w,
            'low_52w': low_52w,
            'price_from_high': price_from_high,
            'price_from_low': price_from_low,
            'current_volatility': current_volatility,
            'avg_volatility': avg_volatility,
            'current_rsi': current_rsi,
            'macd_histogram': macd_histogram,
            'volume_ratio': volume_ratio,
            'trend': trend,
            'trend_score': trend_score,
            'risk_factors': risk_factors,
            'overall_risk': min(overall_risk, 100)
        }

    def create_dashboard(self, symbol, coin_name, output_file):
        """Create comprehensive Plotly dashboard"""
        print(f"\n📊 Creating detailed analyzer for {coin_name}...")

        # Fetch data
        df = self.fetch_ohlcv(symbol, timeframe='1h', days=30)
        if df is None:
            print(f"❌ Failed to fetch data for {symbol}")
            return None

        print(f"✓ Fetched {len(df)} candles")

        # Calculate indicators
        df = self.calculate_indicators(df)
        risk_metrics = self.calculate_risk_factors(df)

        # Create subplots
        fig = make_subplots(
            rows=6, cols=2,
            row_heights=[0.3, 0.15, 0.15, 0.15, 0.15, 0.1],
            column_widths=[0.7, 0.3],
            specs=[
                [{"rowspan": 2}, {"type": "indicator"}],
                [None, {"type": "indicator"}],
                [{"secondary_y": True}, {"type": "indicator"}],
                [{}, {"type": "indicator"}],
                [{}, {"type": "indicator"}],
                [{"colspan": 2}, None]
            ],
            subplot_titles=(
                f'{coin_name}/USDT Price Chart',
                'Current Price',
                '',
                'Risk Score',
                'Volume Analysis',
                'Volatility',
                'RSI Indicator',
                'Volume Ratio',
                'MACD',
                'Trend Score',
                'Risk Factor Breakdown'
            ),
            vertical_spacing=0.05,
            horizontal_spacing=0.1
        )

        # 1. Candlestick chart with EMAs
        fig.add_trace(
            go.Candlestick(
                x=df['datetime'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price',
                increasing_line_color='#3fb950',
                decreasing_line_color='#f85149'
            ),
            row=1, col=1
        )

        # EMAs
        fig.add_trace(go.Scatter(x=df['datetime'], y=df['ema_20'], name='EMA 20',
                                line=dict(color='#58a6ff', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['datetime'], y=df['ema_50'], name='EMA 50',
                                line=dict(color='#f778ba', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['datetime'], y=df['ema_200'], name='EMA 200',
                                line=dict(color='#d29922', width=1)), row=1, col=1)

        # Bollinger Bands
        fig.add_trace(go.Scatter(x=df['datetime'], y=df['bb_upper'], name='BB Upper',
                                line=dict(color='rgba(128,128,128,0.3)', dash='dash'),
                                fill=None), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['datetime'], y=df['bb_lower'], name='BB Lower',
                                line=dict(color='rgba(128,128,128,0.3)', dash='dash'),
                                fill='tonexty', fillcolor='rgba(128,128,128,0.1)'), row=1, col=1)

        # 2. Current Price Indicator
        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=risk_metrics['current_price'],
                delta={'reference': df['close'].iloc[-24], 'relative': True, 'valueformat': '.2%'},
                title={'text': "Current Price (USD)"},
                number={'prefix': "$", 'valueformat': ',.2f'},
                domain={'x': [0, 1], 'y': [0, 1]}
            ),
            row=1, col=2
        )

        # 3. Risk Score Gauge
        risk_color = '#3fb950' if risk_metrics['overall_risk'] < 40 else ('#d29922' if risk_metrics['overall_risk'] < 70 else '#f85149')
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=risk_metrics['overall_risk'],
                title={'text': "Overall Risk"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': risk_color},
                    'steps': [
                        {'range': [0, 40], 'color': "rgba(63, 185, 80, 0.2)"},
                        {'range': [40, 70], 'color': "rgba(210, 153, 34, 0.2)"},
                        {'range': [70, 100], 'color': "rgba(248, 81, 73, 0.2)"}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': risk_metrics['overall_risk']
                    }
                }
            ),
            row=2, col=2
        )

        # 4. Volume Analysis
        colors = ['#3fb950' if row['close'] >= row['open'] else '#f85149' for _, row in df.iterrows()]
        fig.add_trace(
            go.Bar(x=df['datetime'], y=df['volume'], name='Volume',
                  marker_color=colors, opacity=0.7),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['datetime'], y=df['volume_ma'], name='Volume MA',
                      line=dict(color='#58a6ff', width=2)),
            row=3, col=1
        )

        # 5. Volatility Indicator
        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=risk_metrics['current_volatility'],
                delta={'reference': risk_metrics['avg_volatility'], 'relative': False, 'valueformat': '.1f'},
                title={'text': "Volatility %"},
                number={'suffix': "%", 'valueformat': '.1f'}
            ),
            row=3, col=2
        )

        # 6. RSI
        fig.add_trace(
            go.Scatter(x=df['datetime'], y=df['rsi'], name='RSI',
                      line=dict(color='#58a6ff', width=2)),
            row=4, col=1
        )
        # Add horizontal lines for RSI levels
        fig.add_shape(type="line", x0=df['datetime'].iloc[0], x1=df['datetime'].iloc[-1],
                     y0=70, y1=70, line=dict(color="red", dash="dash", width=1),
                     row=4, col=1)
        fig.add_shape(type="line", x0=df['datetime'].iloc[0], x1=df['datetime'].iloc[-1],
                     y0=30, y1=30, line=dict(color="green", dash="dash", width=1),
                     row=4, col=1)
        fig.add_shape(type="line", x0=df['datetime'].iloc[0], x1=df['datetime'].iloc[-1],
                     y0=50, y1=50, line=dict(color="gray", dash="dot", width=1),
                     row=4, col=1)

        # 7. Volume Ratio Indicator
        volume_color = '#3fb950' if risk_metrics['volume_ratio'] > 1 else '#f85149'
        fig.add_trace(
            go.Indicator(
                mode="number+gauge",
                value=risk_metrics['volume_ratio'],
                title={'text': "Vol Ratio"},
                number={'valueformat': '.2f'},
                gauge={
                    'axis': {'range': [0, 3]},
                    'bar': {'color': volume_color},
                    'threshold': {
                        'line': {'color': "white", 'width': 2},
                        'thickness': 0.75,
                        'value': 1
                    }
                }
            ),
            row=4, col=2
        )

        # 8. MACD
        fig.add_trace(
            go.Scatter(x=df['datetime'], y=df['macd'], name='MACD',
                      line=dict(color='#58a6ff', width=2)),
            row=5, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['datetime'], y=df['signal'], name='Signal',
                      line=dict(color='#f778ba', width=2)),
            row=5, col=1
        )
        histogram_colors = ['#3fb950' if x > 0 else '#f85149' for x in df['histogram']]
        fig.add_trace(
            go.Bar(x=df['datetime'], y=df['histogram'], name='Histogram',
                  marker_color=histogram_colors, opacity=0.5),
            row=5, col=1
        )

        # 9. Trend Score
        trend_color = '#3fb950' if risk_metrics['trend_score'] > 60 else ('#d29922' if risk_metrics['trend_score'] > 40 else '#f85149')
        fig.add_trace(
            go.Indicator(
                mode="number+gauge",
                value=risk_metrics['trend_score'],
                title={'text': f"Trend: {risk_metrics['trend']}"},
                number={'valueformat': '.0f'},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': trend_color},
                    'steps': [
                        {'range': [0, 25], 'color': "rgba(248, 81, 73, 0.2)"},
                        {'range': [25, 50], 'color': "rgba(210, 153, 34, 0.2)"},
                        {'range': [50, 75], 'color': "rgba(88, 166, 255, 0.2)"},
                        {'range': [75, 100], 'color': "rgba(63, 185, 80, 0.2)"}
                    ]
                }
            ),
            row=5, col=2
        )

        # 10. Risk Factor Breakdown
        risk_names = ['Volatility', 'RSI', 'Volume', 'Trend']
        risk_values = list(risk_metrics['risk_factors'].values())
        risk_colors_bar = ['#f85149' if v > 15 else ('#d29922' if v > 8 else '#3fb950') for v in risk_values]

        fig.add_trace(
            go.Bar(
                x=risk_names,
                y=risk_values,
                marker_color=risk_colors_bar,
                text=[f"{v:.1f}" for v in risk_values],
                textposition='auto',
                name='Risk Factors'
            ),
            row=6, col=1
        )

        # Update layout
        fig.update_layout(
            title={
                'text': f'{coin_name} Risk Analysis Dashboard<br><sub>Last Updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} Dubai Time</sub>',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 24}
            },
            height=1800,
            showlegend=True,
            template='plotly_dark',
            paper_bgcolor='#0d1117',
            plot_bgcolor='#0d1117',
            font=dict(color='#c9d1d9'),
            hovermode='x unified'
        )

        # Update axes
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')

        # Save to HTML with custom navigation
        html_string = fig.to_html(include_plotlyjs='cdn', full_html=False)

        # Create full HTML with navigation
        full_html = f"""<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{coin_name} Risk Analyzer - Crypto Analytics</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {{
            margin: 0;
            padding: 0;
            background: #0d1117;
            color: #c9d1d9;
        }}
    </style>
</head>
<body>
    <!-- Navigation -->
    <nav class="bg-[#0d1117] border-b border-gray-800 sticky top-0 z-50 backdrop-blur-sm bg-opacity-90">
        <div class="max-w-full mx-auto px-4 sm:px-6 lg:px-8">
            <div class="flex justify-between items-center h-16">
                <div class="flex items-center space-x-3">
                    <div class="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
                        <svg class="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"/>
                        </svg>
                    </div>
                    <span class="text-xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
                        {coin_name} Risk Analyzer
                    </span>
                </div>
                <div class="flex items-center space-x-6">
                    <a href="index.html" class="text-gray-400 hover:text-white transition">← Home</a>
                    <a href="dashboard.html" class="text-gray-400 hover:text-white transition">Dashboard</a>
                    <div class="flex items-center space-x-2">
                        <a href="sol_risk_analyzer.html" class="{'text-blue-400 font-medium' if coin_name == 'SOL' else 'text-gray-400 hover:text-white'} transition">SOL</a>
                        <span class="text-gray-600">|</span>
                        <a href="eth_risk_analyzer.html" class="{'text-blue-400 font-medium' if coin_name == 'ETH' else 'text-gray-400 hover:text-white'} transition">ETH</a>
                        <span class="text-gray-600">|</span>
                        <a href="bnb_risk_analyzer.html" class="{'text-blue-400 font-medium' if coin_name == 'BNB' else 'text-gray-400 hover:text-white'} transition">BNB</a>
                        <span class="text-gray-600">|</span>
                        <a href="dash_risk_analyzer.html" class="{'text-blue-400 font-medium' if coin_name == 'DASH' else 'text-gray-400 hover:text-white'} transition">DASH</a>
                    </div>
                    <div class="flex items-center space-x-2 bg-[#010409] px-3 py-1.5 rounded-lg border border-green-500/30">
                        <div class="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                        <span class="text-green-500 text-sm font-medium">LIVE</span>
                    </div>
                </div>
            </div>
        </div>
    </nav>

    <!-- Plotly Chart -->
    <div style="padding: 20px;">
        {html_string}
    </div>

    <!-- Footer Info -->
    <div class="text-center py-4 text-gray-500 text-sm border-t border-gray-800">
        Auto-updates every 15 minutes | Data: Binance API | Dubai Time (GMT+4)
    </div>
</body>
</html>"""

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(full_html)

        print(f"✅ Dashboard saved: {output_file}")

        # Print summary
        print(f"\n📈 {coin_name} Risk Summary:")
        print(f"  Current Price: ${risk_metrics['current_price']:,.2f}")
        print(f"  Overall Risk: {risk_metrics['overall_risk']:.1f}/100")
        print(f"  Volatility: {risk_metrics['current_volatility']:.1f}%")
        print(f"  RSI: {risk_metrics['current_rsi']:.1f}")
        print(f"  Trend: {risk_metrics['trend']}")
        print(f"  Volume Ratio: {risk_metrics['volume_ratio']:.2f}x")

        return risk_metrics


def main():
    """Generate all risk analyzer dashboards"""
    analyzer = DetailedRiskAnalyzer()

    coins = {
        'SOL': 'SOL/USDT',
        'ETH': 'ETH/USDT',
        'BNB': 'BNB/USDT',
        'DASH': 'DASH/USDT'
    }

    output_dir = "/Users/muhamedalanc/crypto-website"

    print("\n" + "="*70)
    print("  DETAILED RISK ANALYZER - DASHBOARD GENERATOR")
    print("="*70)

    all_metrics = {}

    for coin_name, symbol in coins.items():
        output_file = f"{output_dir}/{coin_name.lower()}_risk_analyzer.html"
        metrics = analyzer.create_dashboard(symbol, coin_name, output_file)
        if metrics:
            all_metrics[coin_name] = metrics

    # Save summary JSON
    summary_file = f"{output_dir}/risk_analyzer_summary.json"
    summary_data = {
        coin: {
            'price': metrics['current_price'],
            'risk': metrics['overall_risk'],
            'volatility': metrics['current_volatility'],
            'rsi': metrics['current_rsi'],
            'trend': metrics['trend']
        }
        for coin, metrics in all_metrics.items()
    }

    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)

    print(f"\n✅ All dashboards generated successfully!")
    print(f"✅ Summary saved: {summary_file}")


if __name__ == "__main__":
    main()
