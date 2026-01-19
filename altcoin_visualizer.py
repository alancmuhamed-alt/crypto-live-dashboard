import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd


class AltcoinRatioVisualizer:
    def __init__(self):
        """Visualizer for Altcoin Ratio analysis."""
        pass

    def create_altcoin_ratio_chart(self, btc_df, ratio_df, support_levels=None,
                                     resistance_levels=None, bsl_ssl=None,
                                     output_file='altcoin_ratio.html'):
        """
        Create Pine Script style chart:
        - BTC candlesticks (gray/transparent)
        - Altcoin Ratio candlesticks (yellow overlay)
        - SMA 50 (orange)
        - EMA 21 (blue) + SMA 20 (green) with fill
        - Auto S/R levels
        - BSL/SSL lines
        - Info table

        Args:
            btc_df: BTC OHLCV data
            ratio_df: Altcoin ratio data with indicators
            support_levels: Support levels DataFrame
            resistance_levels: Resistance levels DataFrame
            bsl_ssl: Dict with BSL and SSL values
            output_file: Output HTML filename
        """
        fig = go.Figure()

        # 1. BTC Candlesticks (background, muted colors)
        fig.add_trace(
            go.Candlestick(
                x=btc_df['datetime'],
                open=btc_df['open'],
                high=btc_df['high'],
                low=btc_df['low'],
                close=btc_df['close'],
                name='BTC/USDT',
                increasing_line_color='rgba(150, 150, 150, 0.3)',
                decreasing_line_color='rgba(100, 100, 100, 0.3)',
                increasing_fillcolor='rgba(150, 150, 150, 0.2)',
                decreasing_fillcolor='rgba(100, 100, 100, 0.2)',
                showlegend=True
            )
        )

        # 2. Altcoin Ratio Candlesticks (yellow overlay)
        fig.add_trace(
            go.Candlestick(
                x=ratio_df['datetime'],
                open=ratio_df['ar_open'],
                high=ratio_df['ar_high'],
                low=ratio_df['ar_low'],
                close=ratio_df['ar_close'],
                name='Alt Ratio [15M]',
                increasing_line_color='rgba(255, 204, 0, 0.6)',
                decreasing_line_color='rgba(255, 204, 0, 0.8)',
                increasing_fillcolor='rgba(255, 204, 0, 0.4)',
                decreasing_fillcolor='rgba(255, 204, 0, 0.6)',
                showlegend=True
            )
        )

        # 3. SMA 50 (orange)
        fig.add_trace(
            go.Scatter(
                x=ratio_df['datetime'],
                y=ratio_df['sma50'],
                name='SMA 50',
                line=dict(color='orange', width=2),
                showlegend=True
            )
        )

        # 4. EMA 21 (blue)
        fig.add_trace(
            go.Scatter(
                x=ratio_df['datetime'],
                y=ratio_df['ema21'],
                name='EMA 21',
                line=dict(color='blue', width=2),
                showlegend=True,
                fill=None
            )
        )

        # 5. SMA 20 (green) with fill to EMA 21
        fig.add_trace(
            go.Scatter(
                x=ratio_df['datetime'],
                y=ratio_df['sma20'],
                name='SMA 20',
                line=dict(color='green', width=2),
                fill='tonexty',
                fillcolor='rgba(0, 255, 255, 0.15)',
                showlegend=True
            )
        )

        # 6. Support Levels (green lines)
        if support_levels is not None and len(support_levels) > 0:
            for idx, level in support_levels.iterrows():
                fig.add_hline(
                    y=level['price'],
                    line_dash="solid",
                    line_color="lime",
                    line_width=2,
                    annotation_text=f"S{idx+1}",
                    annotation_position="right"
                )

        # 7. Resistance Levels (red lines)
        if resistance_levels is not None and len(resistance_levels) > 0:
            for idx, level in resistance_levels.iterrows():
                fig.add_hline(
                    y=level['price'],
                    line_dash="solid",
                    line_color="red",
                    line_width=2,
                    annotation_text=f"R{idx+1}",
                    annotation_position="right"
                )

        # 8. BSL (Buy-Side Liquidity - green)
        if bsl_ssl and 'bsl' in bsl_ssl and bsl_ssl['bsl'] is not None:
            fig.add_hline(
                y=bsl_ssl['bsl'],
                line_dash="dash",
                line_color="green",
                line_width=2,
                annotation_text="BSL",
                annotation_position="left"
            )

        # 9. SSL (Sell-Side Liquidity - red)
        if bsl_ssl and 'ssl' in bsl_ssl and bsl_ssl['ssl'] is not None:
            fig.add_hline(
                y=bsl_ssl['ssl'],
                line_dash="dash",
                line_color="red",
                line_width=2,
                annotation_text="SSL",
                annotation_position="left"
            )

        # Update layout
        fig.update_layout(
            title=dict(
                text='<b>Altcoin Ratio [15M] + BSL/SSL + Auto S/R</b>',
                x=0.5,
                xanchor='center',
                font=dict(size=20)
            ),
            hovermode='x unified',
            template='plotly_dark',
            width=2400,
            height=800,
            showlegend=True,
            xaxis_rangeslider_visible=False,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        fig.update_xaxes(title_text="Time")
        fig.update_yaxes(title_text="Price")

        # Save HTML with interactive config
        config = {
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'eraseshape'],
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'altcoin_ratio',
                'height': 800,
                'width': 2400,
                'scale': 1
            }
        }
        fig.write_html(output_file, config=config)
        print(f"\n✓ Altcoin Ratio chart saved to: {output_file}")

        return output_file

    def create_sr_table_text(self, support_levels, resistance_levels):
        """Create S/R table summary for display."""
        text = "\n" + "="*50
        text += "\n  SUPPORT & RESISTANCE LEVELS"
        text += "\n" + "="*50

        text += "\n\n  RESISTANCES:"
        if resistance_levels is not None and len(resistance_levels) > 0:
            for idx, level in resistance_levels.iterrows():
                # Pine Script metodu: count yok, sadece price var
                text += f"\n    R{idx+1}: ${level['price']:,.2f}"
        else:
            text += "\n    None found"

        text += "\n\n  SUPPORTS:"
        if support_levels is not None and len(support_levels) > 0:
            for idx, level in support_levels.iterrows():
                # Pine Script metodu: count yok, sadece price var
                text += f"\n    S{idx+1}: ${level['price']:,.2f}"
        else:
            text += "\n    None found"

        text += "\n" + "="*50
        return text

    def add_footprint_markers(self, fig, footprint_df):
        """Add footprint markers to chart (aggressive buy/sell/trap)."""
        if footprint_df is None or len(footprint_df) == 0:
            return fig

        # Aggressive buyers (green circles at bottom)
        aggressive_buys = footprint_df[footprint_df['is_aggressive_buy']]
        if len(aggressive_buys) > 0:
            fig.add_trace(
                go.Scatter(
                    x=aggressive_buys['datetime'],
                    y=aggressive_buys['low'],
                    mode='markers',
                    name='Aggressive Buy',
                    marker=dict(
                        symbol='circle',
                        size=8,
                        color='lime',
                        line=dict(color='darkgreen', width=1)
                    ),
                    hovertemplate='<b>Aggressive Buy</b><br>Price: %{y:.2f}<extra></extra>'
                )
            )

        # Aggressive sellers (red circles at top)
        aggressive_sells = footprint_df[footprint_df['is_aggressive_sell']]
        if len(aggressive_sells) > 0:
            fig.add_trace(
                go.Scatter(
                    x=aggressive_sells['datetime'],
                    y=aggressive_sells['high'],
                    mode='markers',
                    name='Aggressive Sell',
                    marker=dict(
                        symbol='circle',
                        size=8,
                        color='red',
                        line=dict(color='darkred', width=1)
                    ),
                    hovertemplate='<b>Aggressive Sell</b><br>Price: %{y:.2f}<extra></extra>'
                )
            )

        # Trap zones (yellow rectangles)
        traps = footprint_df[footprint_df['is_trap']]
        for idx, trap in traps.iterrows():
            fig.add_shape(
                type="rect",
                x0=trap['datetime'] - pd.Timedelta(minutes=7.5),
                x1=trap['datetime'] + pd.Timedelta(minutes=7.5),
                y0=trap['low'],
                y1=trap['high'],
                fillcolor='rgba(255, 255, 0, 0.3)',
                line=dict(color='yellow', width=2)
            )

            # Warning label
            fig.add_annotation(
                x=trap['datetime'],
                y=trap['high'],
                text="⚠",
                showarrow=False,
                font=dict(size=16, color='yellow'),
                bgcolor='rgba(0, 0, 0, 0.5)'
            )

        return fig

    def add_order_flow_profile(self, fig, profile_df, current_time, profile_scale=20, offset=5):
        """Add order flow profile bars to the right side of chart."""
        if profile_df is None or len(profile_df) == 0:
            return fig

        max_vol = max(profile_df['buy_volume'].max(), profile_df['sell_volume'].max())

        if max_vol == 0:
            return fig

        time_offset = pd.Timedelta(minutes=offset * 15)  # 15m timeframe

        # Buy volume bars (green)
        for idx, row in profile_df.iterrows():
            if row['buy_volume'] > 0:
                width = (row['buy_volume'] / max_vol) * profile_scale
                bar_start = current_time + time_offset
                bar_end = bar_start + pd.Timedelta(minutes=width * 15)

                fig.add_shape(
                    type="rect",
                    x0=bar_start,
                    x1=bar_end,
                    y0=row['price_bottom'],
                    y1=row['price_top'],
                    fillcolor='rgba(0, 255, 187, 0.7)',
                    line=dict(width=1, color='rgba(0, 255, 187, 0.7)')
                )

        # Sell volume bars (red, offset to the right)
        sell_offset = time_offset + pd.Timedelta(minutes=(profile_scale + 2) * 15)

        for idx, row in profile_df.iterrows():
            if row['sell_volume'] > 0:
                width = (row['sell_volume'] / max_vol) * profile_scale
                bar_start = current_time + sell_offset
                bar_end = bar_start + pd.Timedelta(minutes=width * 15)

                fig.add_shape(
                    type="rect",
                    x0=bar_start,
                    x1=bar_end,
                    y0=row['price_bottom'],
                    y1=row['price_top'],
                    fillcolor='rgba(255, 17, 0, 0.7)',
                    line=dict(width=1, color='rgba(255, 17, 0, 0.7)')
                )

        return fig

    def create_orderbook_depth_chart(self, orderbook_data, output_file='orderbook_depth.html'):
        """
        Create real-time order book depth chart.

        Args:
            orderbook_data: Order book snapshot with bids and asks
            output_file: Output HTML filename
        """
        if not orderbook_data:
            print("⚠ No order book data available")
            return None

        bids = orderbook_data['bids']  # [[price, qty], ...]
        asks = orderbook_data['asks']

        # Calculate cumulative volumes
        bid_prices = [price for price, _ in bids]
        bid_volumes = [qty for _, qty in bids]
        bid_cumulative = []
        cumsum = 0
        for vol in bid_volumes:
            cumsum += vol
            bid_cumulative.append(cumsum)

        ask_prices = [price for price, _ in asks]
        ask_volumes = [qty for _, qty in asks]
        ask_cumulative = []
        cumsum = 0
        for vol in ask_volumes:
            cumsum += vol
            ask_cumulative.append(cumsum)

        # Create figure
        fig = go.Figure()

        # Bid side (green, left)
        fig.add_trace(
            go.Scatter(
                x=bid_prices,
                y=bid_cumulative,
                name='Bid Depth',
                fill='tozeroy',
                fillcolor='rgba(0, 255, 0, 0.3)',
                line=dict(color='green', width=2),
                hovertemplate='<b>Bid</b><br>Price: $%{x:.2f}<br>Cumulative: %{y:.2f}<extra></extra>'
            )
        )

        # Ask side (red, right)
        fig.add_trace(
            go.Scatter(
                x=ask_prices,
                y=ask_cumulative,
                name='Ask Depth',
                fill='tozeroy',
                fillcolor='rgba(255, 0, 0, 0.3)',
                line=dict(color='red', width=2),
                hovertemplate='<b>Ask</b><br>Price: $%{x:.2f}<br>Cumulative: %{y:.2f}<extra></extra>'
            )
        )

        # Mark best bid/ask
        if bids:
            fig.add_vline(x=bids[0][0], line_dash="dash", line_color="green",
                         annotation_text="Best Bid", annotation_position="top")

        if asks:
            fig.add_vline(x=asks[0][0], line_dash="dash", line_color="red",
                         annotation_text="Best Ask", annotation_position="top")

        # Calculate stats for title
        bid_volume = sum(qty for _, qty in bids)
        ask_volume = sum(qty for _, qty in asks)
        imbalance = bid_volume / ask_volume if ask_volume > 0 else 0
        spread = asks[0][0] - bids[0][0] if asks and bids else 0

        fig.update_layout(
            title=dict(
                text=f'<b>Real-Time Order Book Depth - {orderbook_data["symbol"]}</b><br>' +
                     f'<sub>Spread: ${spread:.2f} | Bid/Ask Ratio: {imbalance:.2f}</sub>',
                x=0.5,
                xanchor='center',
                font=dict(size=18)
            ),
            xaxis_title='Price (USD)',
            yaxis_title='Cumulative Volume',
            template='plotly_dark',
            height=600,
            showlegend=True,
            hovermode='x unified'
        )

        config = {
            'displayModeBar': True,
            'displaylogo': False,
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'orderbook_depth',
                'height': 600,
                'width': 1200,
                'scale': 1
            }
        }
        fig.write_html(output_file, config=config)
        print(f"\n✓ Order book depth chart saved to: {output_file}")

        return output_file

    def add_orderbook_overlay(self, fig, orderbook_stats, y_position='bottom'):
        """
        Add order book statistics overlay to existing chart.

        Args:
            fig: Plotly figure object
            orderbook_stats: Order book statistics dict
            y_position: Position of overlay ('top' or 'bottom')
        """
        if not orderbook_stats:
            return fig

        # Create annotation with order book stats
        stats_text = (
            f"<b>Order Book (Live)</b><br>"
            f"Spread: ${orderbook_stats['spread']:.2f} ({orderbook_stats['spread_pct']:.3f}%)<br>"
            f"Bid Vol: {orderbook_stats['bid_volume']:.2f}<br>"
            f"Ask Vol: {orderbook_stats['ask_volume']:.2f}<br>"
            f"Ratio: {orderbook_stats['volume_ratio']:.2f}<br>"
            f"<b>Imbalance: {orderbook_stats['imbalance']}</b>"
        )

        y_pos = 0.02 if y_position == 'bottom' else 0.98
        y_anchor = 'bottom' if y_position == 'bottom' else 'top'

        fig.add_annotation(
            text=stats_text,
            xref='paper',
            yref='paper',
            x=0.02,
            y=y_pos,
            xanchor='left',
            yanchor=y_anchor,
            showarrow=False,
            bgcolor='rgba(0, 0, 0, 0.7)',
            bordercolor='yellow',
            borderwidth=2,
            font=dict(size=12, color='white')
        )

        return fig

    def create_combined_chart(self, btc_df, ratio_df, orderbook_data,
                               support_levels=None, resistance_levels=None,
                               bsl_ssl=None, footprint_df=None,
                               orderbook_candles=50,
                               output_file='altcoin_combined.html'):
        """
        Create combined multi-panel chart:
        - Top panel: Altcoin Ratio with all indicators
        - Bottom panel: Real-time Order Book Depth

        Args:
            btc_df: BTC OHLCV data
            ratio_df: Altcoin ratio data with indicators
            orderbook_data: Current order book snapshot
            support_levels: Support levels DataFrame
            resistance_levels: Resistance levels DataFrame
            bsl_ssl: Dict with BSL and SSL values
            footprint_df: Footprint analysis data
            orderbook_candles: Number of recent candles to show for order book reference
            output_file: Output HTML filename
        """
        from plotly.subplots import make_subplots

        # Create subplots: 2 rows, 1 column
        # Top panel (70% height): Main chart
        # Bottom panel (30% height): Order book depth
        # SHARED X-AXIS = Alt paneller birlikte hareket eder
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.70, 0.30],
            vertical_spacing=0.05,
            subplot_titles=('<b>Altcoin Ratio [15M] + All Indicators</b>',
                           '<b>Real-Time Order Book Depth</b>'),
            specs=[[{"secondary_y": False}],
                   [{"secondary_y": False}]],
            shared_xaxes=True  # Alt paneller BAĞIMSIZ scroll ETMESİN
        )

        # ========== TOP PANEL: MAIN CHART ==========

        # 1. BTC Candlesticks (background)
        fig.add_trace(
            go.Candlestick(
                x=btc_df['datetime'],
                open=btc_df['open'],
                high=btc_df['high'],
                low=btc_df['low'],
                close=btc_df['close'],
                name='BTC/USDT',
                increasing_line_color='rgba(150, 150, 150, 0.3)',
                decreasing_line_color='rgba(100, 100, 100, 0.3)',
                increasing_fillcolor='rgba(150, 150, 150, 0.2)',
                decreasing_fillcolor='rgba(100, 100, 100, 0.2)',
                showlegend=True
            ),
            row=1, col=1
        )

        # 2. Altcoin Ratio Candlesticks
        fig.add_trace(
            go.Candlestick(
                x=ratio_df['datetime'],
                open=ratio_df['ar_open'],
                high=ratio_df['ar_high'],
                low=ratio_df['ar_low'],
                close=ratio_df['ar_close'],
                name='Alt Ratio [15M]',
                increasing_line_color='rgba(255, 204, 0, 0.6)',
                decreasing_line_color='rgba(255, 204, 0, 0.8)',
                increasing_fillcolor='rgba(255, 204, 0, 0.4)',
                decreasing_fillcolor='rgba(255, 204, 0, 0.6)',
                showlegend=True
            ),
            row=1, col=1
        )

        # 3. SMA 50 (orange)
        fig.add_trace(
            go.Scatter(
                x=ratio_df['datetime'],
                y=ratio_df['sma50'],
                name='SMA 50',
                line=dict(color='orange', width=2),
                showlegend=True
            ),
            row=1, col=1
        )

        # 4. EMA 21 (blue)
        fig.add_trace(
            go.Scatter(
                x=ratio_df['datetime'],
                y=ratio_df['ema21'],
                name='EMA 21',
                line=dict(color='blue', width=2),
                showlegend=True,
                fill=None
            ),
            row=1, col=1
        )

        # 5. SMA 20 (green) with fill
        fig.add_trace(
            go.Scatter(
                x=ratio_df['datetime'],
                y=ratio_df['sma20'],
                name='SMA 20',
                line=dict(color='green', width=2),
                fill='tonexty',
                fillcolor='rgba(0, 255, 255, 0.15)',
                showlegend=True
            ),
            row=1, col=1
        )

        # 6. Support Levels
        if support_levels is not None and len(support_levels) > 0:
            for idx, level in support_levels.iterrows():
                fig.add_hline(
                    y=level['price'],
                    line_dash="solid",
                    line_color="lime",
                    line_width=2,
                    annotation_text=f"S{idx+1}",
                    annotation_position="right",
                    row=1, col=1
                )

        # 7. Resistance Levels
        if resistance_levels is not None and len(resistance_levels) > 0:
            for idx, level in resistance_levels.iterrows():
                fig.add_hline(
                    y=level['price'],
                    line_dash="solid",
                    line_color="red",
                    line_width=2,
                    annotation_text=f"R{idx+1}",
                    annotation_position="right",
                    row=1, col=1
                )

        # 8. BSL (Buy-Side Liquidity)
        if bsl_ssl and 'bsl' in bsl_ssl and bsl_ssl['bsl'] is not None:
            fig.add_hline(
                y=bsl_ssl['bsl'],
                line_dash="dash",
                line_color="green",
                line_width=2,
                annotation_text="BSL",
                annotation_position="left",
                row=1, col=1
            )

        # 9. SSL (Sell-Side Liquidity)
        if bsl_ssl and 'ssl' in bsl_ssl and bsl_ssl['ssl'] is not None:
            fig.add_hline(
                y=bsl_ssl['ssl'],
                line_dash="dash",
                line_color="red",
                line_width=2,
                annotation_text="SSL",
                annotation_position="left",
                row=1, col=1
            )

        # 10. Footprint markers
        if footprint_df is not None and len(footprint_df) > 0:
            # Aggressive buyers
            aggressive_buys = footprint_df[footprint_df['is_aggressive_buy']]
            if len(aggressive_buys) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=aggressive_buys['datetime'],
                        y=aggressive_buys['low'],
                        mode='markers',
                        name='Aggressive Buy',
                        marker=dict(symbol='circle', size=8, color='lime',
                                   line=dict(color='darkgreen', width=1)),
                        hovertemplate='<b>Aggressive Buy</b><br>Price: %{y:.2f}<extra></extra>'
                    ),
                    row=1, col=1
                )

            # Aggressive sellers
            aggressive_sells = footprint_df[footprint_df['is_aggressive_sell']]
            if len(aggressive_sells) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=aggressive_sells['datetime'],
                        y=aggressive_sells['high'],
                        mode='markers',
                        name='Aggressive Sell',
                        marker=dict(symbol='circle', size=8, color='red',
                                   line=dict(color='darkred', width=1)),
                        hovertemplate='<b>Aggressive Sell</b><br>Price: %{y:.2f}<extra></extra>'
                    ),
                    row=1, col=1
                )

        # ========== BOTTOM PANEL: ORDER BOOK DEPTH (ESKİ HALİ) ==========

        if orderbook_data:
            bids = orderbook_data['bids']
            asks = orderbook_data['asks']

            # Calculate cumulative volumes
            bid_prices = [price for price, _ in bids]
            bid_volumes = [qty for _, qty in bids]
            bid_cumulative = []
            cumsum = 0
            for vol in bid_volumes:
                cumsum += vol
                bid_cumulative.append(cumsum)

            ask_prices = [price for price, _ in asks]
            ask_volumes = [qty for _, qty in asks]
            ask_cumulative = []
            cumsum = 0
            for vol in ask_volumes:
                cumsum += vol
                ask_cumulative.append(cumsum)

            # Bid side (green area)
            fig.add_trace(
                go.Scatter(
                    x=bid_prices,
                    y=bid_cumulative,
                    name='Bid Depth',
                    fill='tozeroy',
                    fillcolor='rgba(0, 255, 0, 0.3)',
                    line=dict(color='green', width=2),
                    hovertemplate='<b>Bid</b><br>Price: $%{x:.2f}<br>Cumulative: %{y:.2f}<extra></extra>'
                ),
                row=2, col=1
            )

            # Ask side (red area)
            fig.add_trace(
                go.Scatter(
                    x=ask_prices,
                    y=ask_cumulative,
                    name='Ask Depth',
                    fill='tozeroy',
                    fillcolor='rgba(255, 0, 0, 0.3)',
                    line=dict(color='red', width=2),
                    hovertemplate='<b>Ask</b><br>Price: $%{x:.2f}<br>Cumulative: %{y:.2f}<extra></extra>'
                ),
                row=2, col=1
            )

            # Mark best bid/ask
            if bids:
                fig.add_vline(x=bids[0][0], line_dash="dash", line_color="green",
                             annotation_text="Best Bid", annotation_position="top",
                             row=2, col=1)

            if asks:
                fig.add_vline(x=asks[0][0], line_dash="dash", line_color="red",
                             annotation_text="Best Ask", annotation_position="top",
                             row=2, col=1)

            # Calculate stats
            bid_volume = sum(qty for _, qty in bids)
            ask_volume = sum(qty for _, qty in asks)
            imbalance = bid_volume / ask_volume if ask_volume > 0 else 0
            spread = asks[0][0] - bids[0][0] if asks and bids else 0

            # Add stats annotation
            stats_text = (
                f"Spread: ${spread:.2f} | "
                f"Bid/Ask Ratio: {imbalance:.2f} | "
                f"Symbol: {orderbook_data['symbol']}"
            )

            fig.add_annotation(
                text=stats_text,
                xref='paper',
                yref='paper',
                x=0.5,
                y=0.28,
                xanchor='center',
                yanchor='top',
                showarrow=False,
                bgcolor='rgba(0, 0, 0, 0.7)',
                bordercolor='yellow',
                borderwidth=2,
                font=dict(size=12, color='white')
            )

        # Add timestamp to title
        from datetime import datetime
        import pytz
        dubai_tz = pytz.timezone('Asia/Dubai')
        current_time = datetime.now(dubai_tz).strftime('%H:%M:%S')

        # Update layout - NATIVE ZOOM ALLOWED
        fig.update_layout(
            title=dict(
                text=f'<b>Altcoin Terminal</b><br><sub style="color:#888">Last Update: Dubai {current_time}</sub>',
                x=0.5,
                xanchor='center',
                font=dict(size=16)
            ),
            hovermode='closest',
            template='plotly_dark',
            autosize=True,
            height=None,
            showlegend=True,
            dragmode=False,  # Plotly drag KAPALI - native kullan
            xaxis_rangeslider_visible=False,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.01,
                xanchor="center",
                x=0.5,
                font=dict(size=9)
            ),
            margin=dict(l=30, r=10, t=60, b=30),
            font=dict(size=10),
            hoverlabel=dict(font_size=10)
        )

        # Update axes
        fig.update_xaxes(
            title_text="Time",
            row=1, col=1,
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128, 128, 128, 0.2)'
        )
        fig.update_yaxes(
            title_text="Price",
            row=1, col=1,
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128, 128, 128, 0.2)'
        )
        fig.update_xaxes(
            title_text="Price (USD)",
            row=2, col=1,
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128, 128, 128, 0.2)'
        )
        fig.update_yaxes(
            title_text="Cumulative Volume",
            row=2, col=1,
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128, 128, 128, 0.2)'
        )

        # NATIVE BROWSER ZOOM - Plotly'yi engellemiyoruz
        config = {
            'displayModeBar': False,
            'displaylogo': False,
            'scrollZoom': False,  # Plotly zoom KAPALI - browser kullan
            'doubleClick': False,
            'responsive': True,
            'staticPlot': False,
            'editable': False,
            'showTips': False,
        }

        # HTML oluştur
        html_string = fig.to_html(config=config, include_plotlyjs='cdn')

        # MOBILE TOUCH OPTIMIZED - AUTO FULLSCREEN
        mobile_html = f'''<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, minimum-scale=0.5, maximum-scale=10.0, user-scalable=yes">
    <meta name="mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <title>Altcoin Terminal</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: #0d1117;
            color: #c9d1d9;
            overflow: auto;
            touch-action: auto !important;  /* NATIVE zoom izin ver */
            -webkit-overflow-scrolling: touch;
        }}

        /* NATIVE browser zoom için */
        .plotly-graph-div,
        .js-plotly-plot,
        .plotly,
        svg.main-svg {{
            touch-action: auto !important;  /* Plotly engelini kaldır */
            pointer-events: auto !important;
        }}

        /* Header Bar - HIDDEN on mobile */
        .header {{
            background: linear-gradient(135deg, #1a1f2e 0%, #0d1117 100%);
            padding: 8px 15px;
            box-shadow: 0 2px 20px rgba(0,0,0,0.5);
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            z-index: 9999;
            display: none;  /* HIDDEN - buton yok */
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #30363d;
        }}

        .header-left {{
            display: flex;
            align-items: center;
            gap: 15px;
        }}

        .logo {{
            font-size: 1.5rem;
            font-weight: bold;
            background: linear-gradient(135deg, #58a6ff, #1f6feb);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}

        .status {{
            display: flex;
            align-items: center;
            gap: 8px;
            background: rgba(56, 139, 253, 0.1);
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 0.85rem;
        }}

        .status-dot {{
            width: 8px;
            height: 8px;
            background: #3fb950;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }}

        @keyframes pulse {{
            0%, 100% {{ opacity: 1; transform: scale(1); }}
            50% {{ opacity: 0.6; transform: scale(1.1); }}
        }}

        .controls {{
            display: flex;
            gap: 10px;
        }}

        .btn {{
            background: linear-gradient(135deg, #238636, #2ea043);
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            transition: all 0.3s;
            display: flex;
            align-items: center;
            gap: 6px;
        }}

        .btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(46, 160, 67, 0.4);
        }}

        .btn:active {{
            transform: translateY(0);
        }}

        .btn-fullscreen {{
            background: linear-gradient(135deg, #1f6feb, #388bfd);
        }}

        .btn-fullscreen:hover {{
            box-shadow: 0 4px 12px rgba(56, 139, 253, 0.4);
        }}

        /* Chart Container - FULL SCREEN */
        .chart-container {{
            width: 100vw;
            height: 100vh;
            padding: 0;
            background: #0d1117;
            position: fixed;
            top: 0;
            left: 0;
            overflow: hidden;
        }}

        #chart {{
            width: 100%;
            height: 100%;
            background: #161b22;
            border-radius: 6px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }}

        /* Plotly container responsive */
        #chart .plotly-graph-div {{
            width: 100% !important;
            height: 100% !important;
        }}

        #chart .main-svg {{
            width: 100% !important;
            height: 100% !important;
        }}

        /* Fullscreen Mode */
        .fullscreen-active {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            z-index: 99999;
        }}

        .fullscreen-active .header {{
            background: rgba(13, 17, 23, 0.95);
            backdrop-filter: blur(10px);
        }}

        /* Mobile Optimizations */
        @media (max-width: 768px) {{
            .header {{
                padding: 8px 10px;
            }}

            .logo {{
                font-size: 1rem;
            }}

            .status {{
                font-size: 0.7rem;
                padding: 3px 6px;
            }}

            .status-dot {{
                width: 6px;
                height: 6px;
            }}

            .btn {{
                padding: 5px 10px;
                font-size: 11px;
            }}

            .header-left {{
                gap: 8px;
            }}

            .controls {{
                gap: 5px;
            }}

            .chart-container {{
                padding: 2px;
                height: calc(100vh - 45px);
            }}

            #chart {{
                border-radius: 4px;
            }}
        }}

        /* Extra small screens */
        @media (max-width: 480px) {{
            .logo {{
                font-size: 0.9rem;
            }}

            .status span {{
                display: none;
            }}

            .status-dot {{
                width: 8px;
                height: 8px;
                margin: 0;
            }}

            .btn {{
                padding: 4px 8px;
                font-size: 10px;
            }}

            .chart-container {{
                padding: 1px;
                height: calc(100vh - 40px);
            }}
        }}

        /* Loading Animation */
        .loading {{
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: #58a6ff;
            font-size: 1.2rem;
            display: none;
        }}

        .loading.active {{
            display: block;
        }}
    </style>
</head>
<body>
    <div id="app">
        <!-- Header -->
        <div class="header">
            <div class="header-left">
                <div class="logo">📊 Altcoin Terminal</div>
                <div class="status">
                    <span class="status-dot"></span>
                    <span>Live Data</span>
                </div>
            </div>
            <div class="controls">
                <button class="btn" onclick="resetZoom()" title="Reset Zoom">
                    🔄 Reset
                </button>
                <button class="btn btn-fullscreen" onclick="toggleFullscreen()" id="fullscreenBtn" title="Fullscreen">
                    ⛶ Tam Ekran
                </button>
            </div>
        </div>

        <!-- Chart Container -->
        <div class="chart-container">
            <div id="chart">{html_string[html_string.find('<div>'):html_string.find('</body>')]}</div>
        </div>

        <div class="loading" id="loading">Yükleniyor...</div>
    </div>

    <script>
        // Fullscreen Toggle
        function toggleFullscreen() {{
            const elem = document.documentElement;
            const btn = document.getElementById('fullscreenBtn');

            if (!document.fullscreenElement) {{
                if (elem.requestFullscreen) {{
                    elem.requestFullscreen();
                }} else if (elem.webkitRequestFullscreen) {{
                    elem.webkitRequestFullscreen();
                }} else if (elem.msRequestFullscreen) {{
                    elem.msRequestFullscreen();
                }}
                btn.innerHTML = '⛶ Çıkış';
            }} else {{
                if (document.exitFullscreen) {{
                    document.exitFullscreen();
                }} else if (document.webkitExitFullscreen) {{
                    document.webkitExitFullscreen();
                }} else if (document.msExitFullscreen) {{
                    document.msExitFullscreen();
                }}
                btn.innerHTML = '⛶ Tam Ekran';
            }}
        }}

        // Reset Zoom
        function resetZoom() {{
            const plotDiv = document.querySelector('.plotly-graph-div');
            if (plotDiv) {{
                Plotly.relayout(plotDiv, {{
                    'xaxis.autorange': true,
                    'yaxis.autorange': true,
                    'xaxis2.autorange': true,
                    'yaxis2.autorange': true
                }});
            }}
        }}

        // Update button on fullscreen change
        document.addEventListener('fullscreenchange', function() {{
            const btn = document.getElementById('fullscreenBtn');
            if (!document.fullscreenElement) {{
                btn.innerHTML = '⛶ Tam Ekran';
            }}
        }});

        // Console log
        console.log('%c📊 Altcoin Terminal Pro', 'color: #58a6ff; font-size: 20px; font-weight: bold;');
        console.log('%cMobile-optimized | Zoom/Pan enabled | Fullscreen ready', 'color: #3fb950; font-size: 14px;');

        // Auto-refresh every 15 minutes
        setInterval(function() {{
            console.log('Dashboard yenileniyor...');
            location.reload();
        }}, 900000);

        // NATIVE browser zoom - Plotly'yi bypass et
        if ('ontouchstart' in window) {{
            console.log('📱 NATIVE zoom aktif - elimle yakınlaştır!');

            // Plotly'nin touch olaylarını override et
            setTimeout(function() {{
                const plotDiv = document.querySelector('.plotly-graph-div');
                if (plotDiv) {{
                    // Remove all Plotly touch restrictions
                    plotDiv.style.touchAction = 'auto';

                    const svgs = plotDiv.querySelectorAll('svg');
                    svgs.forEach(svg => {{
                        svg.style.touchAction = 'auto';
                        svg.style.pointerEvents = 'auto';
                    }});

                    console.log('✅ Native zoom enabled!');
                }}
            }}, 500);

            // Auto fullscreen
            setTimeout(function() {{
                if (document.documentElement.requestFullscreen) {{
                    document.documentElement.requestFullscreen().catch(err => {{
                        console.log('Fullscreen:', err.message);
                    }});
                }} else if (document.documentElement.webkitRequestFullscreen) {{
                    document.documentElement.webkitRequestFullscreen();
                }}
            }}, 1000);
        }}

        // Window resize handler
        let resizeTimer;
        window.addEventListener('resize', function() {{
            clearTimeout(resizeTimer);
            resizeTimer = setTimeout(function() {{
                const plotDiv = document.querySelector('.plotly-graph-div');
                if (plotDiv) {{
                    Plotly.Plots.resize(plotDiv);
                }}
            }}, 250);
        }});

        // Orientation change handler
        window.addEventListener('orientationchange', function() {{
            setTimeout(function() {{
                const plotDiv = document.querySelector('.plotly-graph-div');
                if (plotDiv) {{
                    Plotly.Plots.resize(plotDiv);
                }}
            }}, 100);
        }});

        // Initial resize
        window.addEventListener('load', function() {{
            setTimeout(function() {{
                const plotDiv = document.querySelector('.plotly-graph-div');
                if (plotDiv) {{
                    Plotly.Plots.resize(plotDiv);
                }}
            }}, 100);
        }});
    </script>
</body>
</html>'''

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(mobile_html)

        print(f"\n✓ Combined dashboard saved to: {output_file}")

        return output_file
