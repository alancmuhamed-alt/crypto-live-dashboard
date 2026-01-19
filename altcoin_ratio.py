import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import pytz


class AltcoinRatioCalculator:
    def __init__(self):
        """
        Calculate Altcoin Ratio: (TOTAL3 - USDT.D) / BTC.D

        TOTAL3 = Total crypto market cap excluding BTC and ETH
        USDT.D = USDT dominance
        BTC.D = Bitcoin dominance
        """
        # Try multiple exchanges (Binance may be geo-blocked on GitHub Actions)
        self.exchange = self._initialize_exchange()
        self.coingecko_base = "https://api.coingecko.com/api/v3"

    def _initialize_exchange(self):
        """Initialize exchange with fallback options"""
        exchanges_to_try = [
            ('binance', ccxt.binance),
            ('kraken', ccxt.kraken),
            ('bybit', ccxt.bybit),
        ]

        for name, exchange_class in exchanges_to_try:
            try:
                exchange = exchange_class({'enableRateLimit': True})
                # Test if exchange works by loading markets
                exchange.load_markets()
                print(f"✓ Using {name} exchange")
                return exchange
            except Exception as e:
                print(f"⚠ {name} failed: {str(e)[:100]}")
                continue

        # Fallback to binance if all fail (will error later if truly blocked)
        print("⚠ All exchanges failed, using binance as fallback")
        return ccxt.binance({'enableRateLimit': True})

    def fetch_market_caps(self, days=30):
        """
        Fetch market cap data from CoinGecko.

        Returns:
            DataFrame with BTC, ETH, TOTAL market caps
        """
        print("\n⏳ Fetching market cap data from CoinGecko...")

        try:
            # Fetch global market data
            url = f"{self.coingecko_base}/global"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()['data']

            # Current values
            total_market_cap = data['total_market_cap']['usd']
            btc_dominance = data['market_cap_percentage']['btc']
            eth_dominance = data['market_cap_percentage'].get('eth', 0)
            usdt_dominance = data['market_cap_percentage'].get('usdt', 0)

            # Calculate TOTAL3 (excluding BTC and ETH)
            total3 = total_market_cap * (100 - btc_dominance - eth_dominance) / 100

            print(f"✓ Current Market Data:")
            print(f"  Total Market Cap: ${total_market_cap:,.0f}")
            print(f"  BTC Dominance: {btc_dominance:.2f}%")
            print(f"  ETH Dominance: {eth_dominance:.2f}%")
            print(f"  USDT Dominance: {usdt_dominance:.2f}%")
            print(f"  TOTAL3: ${total3:,.0f}")

            return {
                'total_market_cap': total_market_cap,
                'btc_dominance': btc_dominance,
                'eth_dominance': eth_dominance,
                'usdt_dominance': usdt_dominance,
                'total3': total3
            }

        except Exception as e:
            print(f"✗ Error fetching market caps: {e}")
            return None

    def fetch_btc_price_history(self, timeframe='15m', limit=100):
        """Fetch BTC/USDT price data from Binance."""
        try:
            print(f"\n⏳ Fetching BTC/USDT {timeframe} data...")
            ohlcv = self.exchange.fetch_ohlcv('BTC/USDT', timeframe, limit=limit)

            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')

            print(f"✓ Fetched {len(df)} BTC candles")
            return df

        except Exception as e:
            print(f"✗ Error fetching BTC data: {e}")
            return None

    def fetch_overlay_price(self, symbol='ETH/USDT', timeframe='15m', limit=100, days=None):
        """Fetch overlay coin price (e.g., ETH) for scaling.

        Args:
            symbol: Trading pair symbol
            timeframe: Candle timeframe (15m, 1h, etc.)
            limit: Number of candles (ignored if days is provided)
            days: Number of days of historical data to fetch (overrides limit)
        """
        try:
            # If days is specified, calculate required candles
            if days is not None:
                # Calculate candles per day based on timeframe
                timeframe_minutes = {
                    '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
                    '1h': 60, '2h': 120, '4h': 240, '1d': 1440
                }
                minutes = timeframe_minutes.get(timeframe, 15)
                candles_per_day = (24 * 60) // minutes
                limit = days * candles_per_day

            print(f"\n⏳ Fetching {symbol} {timeframe} data for scale factor...")
            print(f"   Requested: {limit} candles (~{limit // 96:.0f} days)")

            # Binance limit is 1000 candles per request
            max_limit = 1000
            all_data = []

            if limit <= max_limit:
                # Single request
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                all_data = ohlcv
            else:
                # Multiple requests needed
                print(f"   Making multiple API calls (Binance limit: {max_limit} candles/request)...")

                # Calculate how many requests we need
                num_requests = (limit + max_limit - 1) // max_limit

                # Fetch data in chunks, working backwards from now
                since = None
                for i in range(num_requests):
                    try:
                        chunk_limit = min(max_limit, limit - len(all_data))
                        ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=chunk_limit, params={'endTime': since} if since else {})

                        if len(ohlcv) == 0:
                            break

                        # Prepend to keep chronological order
                        all_data = ohlcv + all_data

                        # Update since to fetch older data
                        since = ohlcv[0][0] - 1

                        print(f"   Progress: {len(all_data)}/{limit} candles fetched")

                        # Small delay to avoid rate limits
                        if i < num_requests - 1:
                            import time
                            time.sleep(0.5)

                    except Exception as e:
                        print(f"   ⚠ Error in chunk {i+1}: {e}")
                        break

            df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

            # Convert to Dubai timezone (GMT+4)
            dubai_tz = pytz.timezone('Asia/Dubai')
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df['datetime'] = df['datetime'].dt.tz_convert(dubai_tz)

            # Remove duplicates that might occur at boundaries
            df = df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)

            print(f"✓ Fetched {len(df)} {symbol} candles")
            print(f"   Period (Dubai): {df['datetime'].iloc[0]} to {df['datetime'].iloc[-1]}")
            return df

        except Exception as e:
            print(f"✗ Error fetching {symbol} data: {e}")
            import traceback
            traceback.print_exc()
            return None

    def calculate_synthetic_altcoin_ratio(self, overlay_df, current_market_data, symbol_name='ETH'):
        """
        Calculate synthetic altcoin ratio overlay.

        Formula: (TOTAL3 - USDT.D) / BTC.D
        Scale to match overlay coin price (e.g., ETH) - EXACTLY like Pine Script.

        Args:
            overlay_df: Price data of overlay coin (ETH, BTC, etc.)
            current_market_data: Market cap and dominance data
            symbol_name: Name of overlay coin for display
        """
        if current_market_data is None:
            return None

        # Current snapshot values
        total3 = current_market_data['total3']
        btc_dom = current_market_data['btc_dominance']
        usdt_dom = current_market_data['usdt_dominance']

        # Estimate USDT market cap
        total_mc = current_market_data['total_market_cap']
        usdt_mc = total_mc * usdt_dom / 100

        # Calculate raw ratio: (TOTAL3 - USDT.D) / BTC.D
        # This matches Pine Script: raw_c = (t_c - u_c) / b_c
        raw_ratio_current = (total3 - usdt_mc) / btc_dom

        # CRITICAL: Scale factor calculation - EXACTLY like Pine Script
        # var float scale_factor = na
        # if not na(raw_c) and raw_c != 0
        #     scale_factor := close / raw_c
        current_overlay_price = overlay_df['close'].iloc[-1]
        scale_factor = current_overlay_price / raw_ratio_current

        print(f"\n📊 Altcoin Ratio Calculation ({symbol_name} overlay):")
        print(f"  TOTAL3: ${total3:,.0f}")
        print(f"  USDT MC: ${usdt_mc:,.0f}")
        print(f"  BTC Dominance: {btc_dom:.2f}%")
        print(f"  Raw Ratio (current): {raw_ratio_current:,.2f}")
        print(f"  {symbol_name} Price: ${current_overlay_price:,.2f}")
        print(f"  Scale Factor: {scale_factor:.8f}")

        # Create ratio OHLC by applying the same transformation to each candle
        ratio_df = overlay_df.copy()

        # For each candle, calculate raw ratio from overlay price movements
        # Then apply scale factor
        for idx in range(len(ratio_df)):
            # Simplified: assume ratio follows overlay price movements
            # In real implementation, you'd need historical TOTAL3, USDT.D, BTC.D data
            overlay_ratio = ratio_df.iloc[idx]['open'] / current_overlay_price
            ratio_df.loc[ratio_df.index[idx], 'ar_open'] = raw_ratio_current * overlay_ratio * scale_factor

            overlay_ratio = ratio_df.iloc[idx]['high'] / current_overlay_price
            ratio_df.loc[ratio_df.index[idx], 'ar_high'] = raw_ratio_current * overlay_ratio * scale_factor

            overlay_ratio = ratio_df.iloc[idx]['low'] / current_overlay_price
            ratio_df.loc[ratio_df.index[idx], 'ar_low'] = raw_ratio_current * overlay_ratio * scale_factor

            overlay_ratio = ratio_df.iloc[idx]['close'] / current_overlay_price
            ratio_df.loc[ratio_df.index[idx], 'ar_close'] = raw_ratio_current * overlay_ratio * scale_factor

        return ratio_df

    def calculate_indicators(self, df):
        """
        Calculate technical indicators matching Pine Script.

        - SMA 50
        - EMA 21
        - SMA 20
        - Bull Band (EMA21 + SMA20 fill)
        """
        # SMA 50
        df['sma50'] = df['ar_close'].rolling(window=50).mean()

        # EMA 21
        df['ema21'] = df['ar_close'].ewm(span=21, adjust=False).mean()

        # SMA 20
        df['sma20'] = df['ar_close'].rolling(window=20).mean()

        return df

    def find_auto_sr_levels_pinescript(self, df, left_right=3, tolerance=30.0, max_supports=4, max_resistances=4):
        """
        Pine Script AUTO S/R mantığını birebir uygula.

        Pine Script mantığı:
        1. pivothigh/pivotlow ile pivot noktalarını bul
        2. Her yeni pivot için, mevcut seviyelere tolerance içinde bir şey var mı kontrol et
        3. Yoksa, en son 6 seviyeye ekle (queue mantığı - en eskisi düşer)

        Args:
            df: DataFrame with OHLC data
            left_right: Pivot window (leftRight parametresi)
            tolerance: Tolerance for level grouping
            max_supports: Maksimum support sayısı
            max_resistances: Maksimum resistance sayısı

        Returns:
            Dict with 'supports' and 'resistances' DataFrames
        """
        # Pine Script'teki gibi değişkenler
        supports = []  # [s1, s2, s3, s4, s5, s6] mantığı
        resistances = []  # [r1, r2, r3, r4, r5, r6] mantığı

        max_levels = 6  # Pine Script'te sabit 6

        # DataFrame'i iterasyonla geç (Pine Script bar_index mantığı)
        for i in range(left_right, len(df) - left_right):
            # PIVOT HIGH kontrolü (resistance)
            is_pivot_high = True
            current_high = df['ar_high'].iloc[i]

            for j in range(i - left_right, i + left_right + 1):
                if j == i:
                    continue
                if df['ar_high'].iloc[j] >= current_high:
                    is_pivot_high = False
                    break

            if is_pivot_high:
                yR = current_high
                nearR = False

                # Mevcut seviyelere tolerance içinde bir şey var mı?
                for r_level in resistances:
                    if abs(r_level['price'] - yR) <= tolerance:
                        nearR = True
                        break

                # Yoksa yeni seviye ekle
                if not nearR:
                    new_resistance = {
                        'price': yR,
                        'index': i,
                        'datetime': df['datetime'].iloc[i],
                        'type': 'resistance'
                    }

                    # Pine Script mantığı: en başa ekle, sondan sil
                    resistances.insert(0, new_resistance)
                    if len(resistances) > max_levels:
                        resistances.pop()

            # PIVOT LOW kontrolü (support)
            is_pivot_low = True
            current_low = df['ar_low'].iloc[i]

            for j in range(i - left_right, i + left_right + 1):
                if j == i:
                    continue
                if df['ar_low'].iloc[j] <= current_low:
                    is_pivot_low = False
                    break

            if is_pivot_low:
                yS = current_low
                nearS = False

                # Mevcut seviyelere tolerance içinde bir şey var mı?
                for s_level in supports:
                    if abs(s_level['price'] - yS) <= tolerance:
                        nearS = True
                        break

                # Yoksa yeni seviye ekle
                if not nearS:
                    new_support = {
                        'price': yS,
                        'index': i,
                        'datetime': df['datetime'].iloc[i],
                        'type': 'support'
                    }

                    # Pine Script mantığı: en başa ekle, sondan sil
                    supports.insert(0, new_support)
                    if len(supports) > max_levels:
                        supports.pop()

        # En son N tanesini al (kullanıcı ayarı)
        final_supports = supports[:max_supports]
        final_resistances = resistances[:max_resistances]

        # DataFrame'e çevir
        supports_df = pd.DataFrame(final_supports) if final_supports else pd.DataFrame()
        resistances_df = pd.DataFrame(final_resistances) if final_resistances else pd.DataFrame()

        # Sıralama: Resistance'lar yüksekten alçağa, Support'lar alçaktan yükseğe
        if len(resistances_df) > 0:
            resistances_df = resistances_df.sort_values('price', ascending=False).reset_index(drop=True)

        if len(supports_df) > 0:
            supports_df = supports_df.sort_values('price', ascending=True).reset_index(drop=True)

        return {
            'supports': supports_df,
            'resistances': resistances_df
        }

    def cluster_support_resistance(self, pivots, tolerance=30, max_levels=4):
        """
        Cluster nearby pivot points into S/R levels.

        Args:
            pivots: DataFrame with pivot points
            tolerance: Price tolerance for clustering
            max_levels: Maximum number of levels to return
        """
        if len(pivots) == 0:
            return []

        levels = []
        used_indices = set()

        for idx, pivot in pivots.iterrows():
            if idx in used_indices:
                continue

            price = pivot['price']
            cluster = [price]
            used_indices.add(idx)

            # Find nearby pivots within tolerance
            for idx2, pivot2 in pivots.iterrows():
                if idx2 in used_indices:
                    continue
                if abs(pivot2['price'] - price) <= tolerance:
                    cluster.append(pivot2['price'])
                    used_indices.add(idx2)

            # Average of cluster
            level_price = np.mean(cluster)
            levels.append({
                'price': level_price,
                'count': len(cluster),
                'type': pivot['type']
            })

        # Sort by count (strength) and take top N
        levels_df = pd.DataFrame(levels)
        if len(levels_df) > 0:
            levels_df = levels_df.sort_values('count', ascending=False).head(max_levels)
            levels_df = levels_df.sort_values('price')

        return levels_df

    def calculate_footprint(self, df, lookback=50, imbalance_threshold=2.0):
        """
        Calculate footprint analysis - aggressive buy/sell detection.

        Args:
            df: DataFrame with OHLC data
            lookback: Number of candles to analyze
            imbalance_threshold: Ratio threshold for trap detection
        """
        footprint_data = []

        for i in range(min(lookback, len(df))):
            candle = df.iloc[-(i+1)]

            candle_open = candle['ar_open']
            candle_close = candle['ar_close']
            candle_high = candle['ar_high']
            candle_low = candle['ar_low']
            candle_range = candle_high - candle_low

            if candle_range > 0:
                # Aggressive buy/sell estimation
                aggressive_buy = candle_range if candle_close > candle_open else 0
                aggressive_sell = candle_range if candle_close < candle_open else 0

                buy_ratio = aggressive_buy / candle_range
                sell_ratio = aggressive_sell / candle_range

                # Imbalance detection
                if buy_ratio > 0 and sell_ratio > 0:
                    imbalance_ratio = max(buy_ratio / sell_ratio, sell_ratio / buy_ratio)
                else:
                    imbalance_ratio = 0

                is_trap = imbalance_ratio > imbalance_threshold

                footprint_data.append({
                    'datetime': candle['datetime'],
                    'index': len(df) - i - 1,
                    'high': candle_high,
                    'low': candle_low,
                    'buy_ratio': buy_ratio,
                    'sell_ratio': sell_ratio,
                    'is_aggressive_buy': buy_ratio > 0.6,
                    'is_aggressive_sell': sell_ratio > 0.6,
                    'is_trap': is_trap,
                    'imbalance_ratio': imbalance_ratio
                })

        return pd.DataFrame(footprint_data)

    def calculate_order_flow_profile(self, df, lookback=20, resolution=15):
        """
        Calculate Order Flow Profile (volume distribution by price level).

        Args:
            df: DataFrame with OHLC data
            lookback: Number of candles to analyze
            resolution: Number of price bins
        """
        recent_data = df.tail(lookback)

        if len(recent_data) == 0:
            return None

        max_price = recent_data['ar_high'].max()
        min_price = recent_data['ar_low'].min()
        price_step = (max_price - min_price) / resolution

        buy_bins = np.zeros(resolution)
        sell_bins = np.zeros(resolution)

        for idx, candle in recent_data.iterrows():
            candle_high = candle['ar_high']
            candle_low = candle['ar_low']
            candle_range = candle_high - candle_low

            # Estimate buy/sell volume
            buy_vol = candle_range if candle['ar_close'] > candle['ar_open'] else 0
            sell_vol = candle_range if candle['ar_close'] < candle['ar_open'] else 0

            # Distribute volume to bins
            for i in range(resolution):
                bin_bottom = min_price + (i * price_step)
                bin_top = min_price + ((i + 1) * price_step)

                # Check if candle intersects this bin
                candle_in_bin = not (candle_low > bin_top or candle_high < bin_bottom)

                if candle_in_bin:
                    buy_bins[i] += buy_vol
                    sell_bins[i] += sell_vol

        profile_data = []
        for i in range(resolution):
            bin_bottom = min_price + (i * price_step)
            bin_top = min_price + ((i + 1) * price_step)

            profile_data.append({
                'bin_index': i,
                'price_bottom': bin_bottom,
                'price_top': bin_top,
                'price_mid': (bin_bottom + bin_top) / 2,
                'buy_volume': buy_bins[i],
                'sell_volume': sell_bins[i],
                'net_volume': buy_bins[i] - sell_bins[i],
                'total_volume': buy_bins[i] + sell_bins[i]
            })

        return pd.DataFrame(profile_data)
