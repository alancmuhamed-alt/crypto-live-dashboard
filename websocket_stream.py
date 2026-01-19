import websocket
import json
import threading
import time
from datetime import datetime
import pandas as pd


class BinanceWebSocketStream:
    def __init__(self, symbol='ethusdt', preload_history=True, days=10):
        """
        Real-time Binance WebSocket stream.

        Args:
            symbol: Trading pair in Binance format (lowercase, no slash)
            preload_history: Fetch historical data on init
            days: Number of days of historical data to preload
        """
        self.symbol = symbol.lower()
        self.ws_url = f"wss://stream.binance.com:9443/ws/{self.symbol}@kline_15m"
        self.ws = None
        self.is_running = False
        self.current_candle = None
        self.candle_history = []
        self.callbacks = []
        self.lock = threading.Lock()
        self.preload_history = preload_history
        self.days = days

    def on_message(self, ws, message):
        """Handle incoming WebSocket messages."""
        try:
            data = json.loads(message)

            if 'k' in data:
                kline = data['k']

                candle_data = {
                    'timestamp': kline['t'],
                    'datetime': datetime.fromtimestamp(kline['t'] / 1000),
                    'open': float(kline['o']),
                    'high': float(kline['h']),
                    'low': float(kline['l']),
                    'close': float(kline['c']),
                    'volume': float(kline['v']),
                    'is_closed': kline['x'],
                    'symbol': self.symbol.upper()
                }

                with self.lock:
                    self.current_candle = candle_data

                    # If candle closed, add to history (avoid duplicates)
                    if candle_data['is_closed']:
                        # Check if already in history
                        existing = [c for c in self.candle_history if c['timestamp'] == candle_data['timestamp']]
                        if not existing:
                            self.candle_history.append(candle_data)
                            print(f"🟢 [{candle_data['datetime']}] {self.symbol.upper()} Candle Closed: "
                                  f"O:{candle_data['open']:.2f} H:{candle_data['high']:.2f} "
                                  f"L:{candle_data['low']:.2f} C:{candle_data['close']:.2f}")

                # Trigger callbacks
                for callback in self.callbacks:
                    callback(candle_data)

        except Exception as e:
            print(f"✗ Error processing message: {e}")

    def on_error(self, ws, error):
        """Handle WebSocket errors."""
        print(f"✗ WebSocket error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket close - AUTO RECONNECT."""
        print(f"⚠ WebSocket closed: {close_status_code} - {close_msg}")
        self.is_running = False

        # AUTO RECONNECT after 5 seconds
        print(f"🔄 Reconnecting in 5 seconds...")
        time.sleep(5)
        self.start()

    def on_open(self, ws):
        """Handle WebSocket open."""
        print(f"✓ WebSocket connected: {self.symbol.upper()} 15m stream")
        self.is_running = True

    def load_historical_data(self):
        """Load historical candle data from Binance REST API."""
        try:
            import ccxt
            exchange = ccxt.binance()

            # Convert symbol format (ethusdt -> ETH/USDT)
            symbol_formatted = f"{self.symbol[:3].upper()}/{self.symbol[3:].upper()}"

            limit = self.days * 96  # 96 candles per day for 15m
            print(f"📥 Preloading {limit} historical candles for {symbol_formatted}...")

            ohlcv = exchange.fetch_ohlcv(symbol_formatted, '15m', limit=limit)

            historical_candles = []
            for candle in ohlcv:
                candle_data = {
                    'timestamp': candle[0],
                    'datetime': datetime.fromtimestamp(candle[0] / 1000),
                    'open': candle[1],
                    'high': candle[2],
                    'low': candle[3],
                    'close': candle[4],
                    'volume': candle[5],
                    'is_closed': True,  # Historical candles are all closed
                    'symbol': self.symbol.upper()
                }
                historical_candles.append(candle_data)

            with self.lock:
                self.candle_history = historical_candles

            print(f"✓ Loaded {len(historical_candles)} historical candles")
            return True

        except Exception as e:
            print(f"⚠ Failed to load historical data: {e}")
            return False

    def start(self):
        """Start WebSocket connection in background thread."""
        # Load historical data first
        if self.preload_history:
            self.load_historical_data()

        def run_websocket():
            self.ws = websocket.WebSocketApp(
                self.ws_url,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close,
                on_open=self.on_open
            )
            self.ws.run_forever()

        thread = threading.Thread(target=run_websocket, daemon=True)
        thread.start()

        # Wait for connection
        time.sleep(2)

    def stop(self):
        """Stop WebSocket connection."""
        if self.ws:
            self.ws.close()
        self.is_running = False
        print(f"⚠ WebSocket stopped for {self.symbol.upper()}")

    def get_current_candle(self):
        """Get current (live) candle data."""
        with self.lock:
            return self.current_candle

    def get_candle_history(self):
        """Get closed candles history."""
        with self.lock:
            return self.candle_history.copy()

    def get_dataframe(self, limit=100, include_current=True):
        """
        Convert history to pandas DataFrame.

        Args:
            limit: Number of candles to return
            include_current: Include the currently forming candle (default True)
        """
        with self.lock:
            history = self.candle_history.copy()
            current = self.current_candle

        # ŞU ANKİ CANLI MUMU EKLE
        if include_current and current is not None:
            # Current candle'ı ekle (henüz kapanmamış)
            history.append(current)

        if len(history) == 0:
            return pd.DataFrame()

        df = pd.DataFrame(history)
        df = df.tail(limit)

        # Datetime'ı timezone-aware yap (Dubai)
        import pytz
        dubai_tz = pytz.timezone('Asia/Dubai')
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df['datetime'] = df['datetime'].dt.tz_convert(dubai_tz)

        return df[['datetime', 'open', 'high', 'low', 'close', 'volume']]

    def add_callback(self, callback):
        """
        Add callback function to be called on each candle update.

        Args:
            callback: Function that takes candle_data dict as argument
        """
        self.callbacks.append(callback)

    def wait_for_candle(self, timeout=30):
        """Wait for at least one candle to arrive."""
        start_time = time.time()

        while time.time() - start_time < timeout:
            if self.current_candle is not None:
                return True
            time.sleep(0.5)

        return False


class BinanceOrderBookStream:
    def __init__(self, symbol='ethusdt', depth_level=20):
        """
        Real-time Binance Order Book Depth stream.

        Args:
            symbol: Trading pair (lowercase)
            depth_level: Number of levels (5, 10, or 20)
        """
        self.symbol = symbol.lower()
        self.depth_level = depth_level
        self.ws_url = f"wss://stream.binance.com:9443/ws/{self.symbol}@depth{depth_level}@100ms"
        self.ws = None
        self.is_running = False
        self.current_orderbook = None
        self.orderbook_history = []
        self.callbacks = []
        self.lock = threading.Lock()

    def on_message(self, ws, message):
        """Handle incoming order book messages."""
        try:
            data = json.loads(message)

            if 'bids' in data and 'asks' in data:
                # Parse bids and asks
                bids = [[float(price), float(qty)] for price, qty in data['bids']]
                asks = [[float(price), float(qty)] for price, qty in data['asks']]

                orderbook_data = {
                    'timestamp': data.get('lastUpdateId', int(time.time() * 1000)),
                    'datetime': datetime.now(),
                    'bids': bids,  # [[price, quantity], ...]
                    'asks': asks,
                    'symbol': self.symbol.upper()
                }

                with self.lock:
                    self.current_orderbook = orderbook_data
                    self.orderbook_history.append(orderbook_data)

                    # Keep only last 100 snapshots
                    if len(self.orderbook_history) > 100:
                        self.orderbook_history.pop(0)

                # Trigger callbacks
                for callback in self.callbacks:
                    callback(orderbook_data)

        except Exception as e:
            print(f"✗ Error processing order book: {e}")

    def on_error(self, ws, error):
        """Handle WebSocket errors."""
        print(f"✗ Order book WebSocket error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket close."""
        print(f"⚠ Order book WebSocket closed")
        self.is_running = False

    def on_open(self, ws):
        """Handle WebSocket open."""
        print(f"✓ Order Book WebSocket connected: {self.symbol.upper()} depth@{self.depth_level}")
        self.is_running = True

    def start(self):
        """Start WebSocket connection."""
        def run_websocket():
            self.ws = websocket.WebSocketApp(
                self.ws_url,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close,
                on_open=self.on_open
            )
            self.ws.run_forever()

        thread = threading.Thread(target=run_websocket, daemon=True)
        thread.start()
        time.sleep(2)

    def stop(self):
        """Stop WebSocket connection."""
        if self.ws:
            self.ws.close()
        self.is_running = False
        print(f"⚠ Order Book WebSocket stopped for {self.symbol.upper()}")

    def get_current_orderbook(self):
        """Get current order book snapshot."""
        with self.lock:
            return self.current_orderbook

    def get_orderbook_stats(self):
        """Calculate order book statistics."""
        with self.lock:
            if not self.current_orderbook:
                return None

            bids = self.current_orderbook['bids']
            asks = self.current_orderbook['asks']

            # Calculate total volume
            bid_volume = sum(qty for _, qty in bids)
            ask_volume = sum(qty for _, qty in asks)

            # Calculate weighted average prices
            bid_wavg = sum(price * qty for price, qty in bids) / bid_volume if bid_volume > 0 else 0
            ask_wavg = sum(price * qty for price, qty in asks) / ask_volume if ask_volume > 0 else 0

            # Best bid/ask
            best_bid = bids[0][0] if bids else 0
            best_ask = asks[0][0] if asks else 0

            # Spread
            spread = best_ask - best_bid
            spread_pct = (spread / best_bid * 100) if best_bid > 0 else 0

            # Buy/Sell ratio
            volume_ratio = bid_volume / ask_volume if ask_volume > 0 else 0

            return {
                'best_bid': best_bid,
                'best_ask': best_ask,
                'spread': spread,
                'spread_pct': spread_pct,
                'bid_volume': bid_volume,
                'ask_volume': ask_volume,
                'volume_ratio': volume_ratio,
                'bid_wavg': bid_wavg,
                'ask_wavg': ask_wavg,
                'imbalance': 'BUY' if volume_ratio > 1.2 else 'SELL' if volume_ratio < 0.8 else 'NEUTRAL'
            }

    def add_callback(self, callback):
        """Add callback for order book updates."""
        self.callbacks.append(callback)

    def wait_for_orderbook(self, timeout=10):
        """Wait for first order book snapshot."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.current_orderbook is not None:
                return True
            time.sleep(0.5)
        return False


class MultiStreamManager:
    def __init__(self):
        """Manage multiple WebSocket streams."""
        self.streams = {}
        self.orderbook_streams = {}

    def add_stream(self, symbol):
        """Add a new candle stream."""
        if symbol not in self.streams:
            stream = BinanceWebSocketStream(symbol)
            self.streams[symbol] = stream
            stream.start()
            return stream
        return self.streams[symbol]

    def add_orderbook_stream(self, symbol, depth_level=20):
        """Add a new order book stream."""
        key = f"{symbol}_depth"
        if key not in self.orderbook_streams:
            stream = BinanceOrderBookStream(symbol, depth_level)
            self.orderbook_streams[key] = stream
            stream.start()
            return stream
        return self.orderbook_streams[key]

    def get_stream(self, symbol):
        """Get existing candle stream."""
        return self.streams.get(symbol)

    def get_orderbook_stream(self, symbol):
        """Get existing order book stream."""
        key = f"{symbol}_depth"
        return self.orderbook_streams.get(key)

    def stop_all(self):
        """Stop all streams."""
        for symbol, stream in self.streams.items():
            stream.stop()
        for key, stream in self.orderbook_streams.items():
            stream.stop()
        self.streams.clear()
        self.orderbook_streams.clear()
        print("⚠ All WebSocket streams stopped")
