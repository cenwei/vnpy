""""""

import importlib
import json
import logging
import os
import sys
import traceback
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Callable
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from copy import copy
from vnpy.trader.setting import SETTINGS
from tzlocal import get_localzone
from functools import lru_cache

from vnpy.event import Event, EventEngine
from vnpy.trader.engine import BaseEngine, MainEngine
from vnpy.trader.object import (
    OrderRequest,
    SubscribeRequest,
    HistoryRequest,
    LogData,
    TickData,
    BarData,
    ContractData
)
from vnpy.trader.event import (
    EVENT_TICK,
    EVENT_ORDER,
    EVENT_TRADE,
    EVENT_POSITION
)
from vnpy.trader.constant import (
    Direction,
    OrderType,
    Interval,
    Exchange,
    Offset,
    Status
)
from vnpy.trader.utility import TRADER_DIR, get_folder_path, load_json, save_json, extract_vt_symbol, round_to
from vnpy.trader.util_logger import setup_logger
from vnpy.trader.database import database_manager
from vnpy.trader.rqdata import rqdata_client
from vnpy.trader.converter import OffsetConverter
from vnpy.component.cta_position import CtaPosition

from .base import (
    APP_NAME,
    EVENT_CTA_LOG,
    EVENT_CTA_STRATEGY,
    EVENT_CTA_STOPORDER,
    EngineType,
    StopOrder,
    StopOrderStatus,
    STOPORDER_PREFIX
)
from .template import CtaTemplate


STOP_STATUS_MAP = {
    Status.SUBMITTING: StopOrderStatus.WAITING,
    Status.NOTTRADED: StopOrderStatus.WAITING,
    Status.PARTTRADED: StopOrderStatus.TRIGGERED,
    Status.ALLTRADED: StopOrderStatus.TRIGGERED,
    Status.CANCELLED: StopOrderStatus.CANCELLED,
    Status.REJECTED: StopOrderStatus.CANCELLED
}


class CtaEngine(BaseEngine):
    """"""

    engine_type = EngineType.LIVE  # live trading engine

    setting_filename = "cta_strategy_setting.json"
    data_filename = "cta_strategy_data.json"

    # 引擎配置文件
    engine_filename = "cta_strategy_config.json"

    def __init__(self, main_engine: MainEngine, event_engine: EventEngine):
        """"""
        super(CtaEngine, self).__init__(
            main_engine, event_engine, APP_NAME)

        self.strategy_setting = {}  # strategy_name: dict
        self.strategy_data = {}     # strategy_name: dict

        self.classes = {}           # class_name: stategy_class
        self.strategies = {}        # strategy_name: strategy
        self.engine_config = {}

        # Strategy pos dict,key:strategy instance name, value: pos dict
        self.strategy_pos_dict = {}
        self.strategy_loggers = {}  # strategy_name: logger

        self.symbol_strategy_map = defaultdict(
            list)                   # vt_symbol: strategy list
        self.orderid_strategy_map = {}  # vt_orderid: strategy
        self.strategy_orderid_map = defaultdict(
            set)                    # strategy_name: orderid list

        self.stop_order_count = 0   # for generating stop_orderid
        self.stop_orders = {}       # stop_orderid: stop_order

        self.init_executor = ThreadPoolExecutor(max_workers=1)

        self.rq_client = None
        self.rq_symbols = set()

        self.vt_tradeids = set()    # for filtering duplicate trade

        self.offset_converter = OffsetConverter(self.main_engine)

        self.positions = {}

    def init_engine(self):
        """
        """
        self.init_rqdata()
        self.load_strategy_class()
        self.load_strategy_setting()
        self.load_strategy_data()

        self.register_event()
        self.register_funcs()
        self.write_log("CTA策略引擎初始化成功")

    def close(self):
        """"""
        self.stop_all_strategies()

        # 保存引擎配置
        save_json(self.engine_filename, self.engine_config)

    def register_event(self):
        """"""
        self.event_engine.register(EVENT_TICK, self.process_tick_event)
        self.event_engine.register(EVENT_ORDER, self.process_order_event)
        self.event_engine.register(EVENT_TRADE, self.process_trade_event)
        self.event_engine.register(EVENT_POSITION, self.process_position_event)

    def register_funcs(self):
        """
        register the funcs to main_engine
        :return:
        """
        self.main_engine.get_strategy_status = self.get_strategy_status
        self.main_engine.get_strategy_pos = self.get_strategy_pos
        self.main_engine.compare_pos = self.compare_pos
        self.main_engine.add_strategy = self.add_strategy
        self.main_engine.init_strategy = self.init_strategy
        self.main_engine.start_strategy = self.start_strategy
        self.main_engine.stop_strategy = self.stop_strategy
        self.main_engine.remove_strategy = self.remove_strategy
        self.main_engine.reload_strategy = self.reload_strategy

        # 注册到远程服务调用
        if self.main_engine.rpc_service:
            self.main_engine.rpc_service.register(self.main_engine.get_strategy_status)
            self.main_engine.rpc_service.register(self.main_engine.get_strategy_pos)
            self.main_engine.rpc_service.register(self.main_engine.compare_pos)
            self.main_engine.rpc_service.register(self.main_engine.add_strategy)
            self.main_engine.rpc_service.register(self.main_engine.init_strategy)
            self.main_engine.rpc_service.register(self.main_engine.start_strategy)
            self.main_engine.rpc_service.register(self.main_engine.stop_strategy)
            self.main_engine.rpc_service.register(self.main_engine.remove_strategy)
            self.main_engine.rpc_service.register(self.main_engine.reload_strategy)
            # self.main_engine.rpc_service.register(self.main_engine.save_strategy_data)
            # self.main_engine.rpc_service.register(self.main_engine.save_strategy_snapshot)
            # self.main_engine.rpc_service.register(self.main_engine.clean_strategy_cache)

    def init_rqdata(self):
        """
        Init RQData client.
        """
        username = SETTINGS["rqdata.username"]
        if username:
            result = rqdata_client.init()
            if result:
                self.write_log("RQData数据接口初始化成功")

    def query_bar_from_rq(
        self, symbol: str, exchange: Exchange, interval: Interval, start: datetime, end: datetime
    ):
        """
        Query bar data from RQData.
        """
        req = HistoryRequest(
            symbol=symbol,
            exchange=exchange,
            interval=interval,
            start=start,
            end=end
        )
        data = rqdata_client.query_history(req)
        return data

    def process_tick_event(self, event: Event):
        """"""
        tick = event.data

        strategies = self.symbol_strategy_map[tick.vt_symbol]
        if not strategies:
            return

        self.check_stop_order(tick)

        for strategy in strategies:
            if strategy.inited:
                self.call_strategy_func(strategy, strategy.on_tick, tick)

    def process_order_event(self, event: Event):
        """"""
        order = event.data

        self.offset_converter.update_order(order)

        strategy = self.orderid_strategy_map.get(order.vt_orderid, None)
        if not strategy:
            return

        # Remove vt_orderid if order is no longer active.
        vt_orderids = self.strategy_orderid_map[strategy.strategy_name]
        if order.vt_orderid in vt_orderids and not order.is_active():
            vt_orderids.remove(order.vt_orderid)

        # For server stop order, call strategy on_stop_order function
        if order.type == OrderType.STOP:
            so = StopOrder(
                vt_symbol=order.vt_symbol,
                direction=order.direction,
                offset=order.offset,
                price=order.price,
                volume=order.volume,
                stop_orderid=order.vt_orderid,
                strategy_name=strategy.strategy_name,
                status=STOP_STATUS_MAP[order.status],
                vt_orderids=[order.vt_orderid],
            )
            self.call_strategy_func(strategy, strategy.on_stop_order, so)

        # Call strategy on_order function
        self.call_strategy_func(strategy, strategy.on_order, order)

    def process_trade_event(self, event: Event):
        """"""
        trade = event.data

        # Filter duplicate trade push
        if trade.vt_tradeid in self.vt_tradeids:
            return
        self.vt_tradeids.add(trade.vt_tradeid)

        self.offset_converter.update_trade(trade)

        strategy = self.orderid_strategy_map.get(trade.vt_orderid, None)
        if not strategy:
            return

        contract = self.main_engine.get_contract(strategy.vt_symbol)
        if not contract:
            volume = trade.volume
        else:
            # Round order price and volume to nearest incremental value
            volume = round_to(trade.volume, contract.min_volume)

        # Update strategy pos before calling on_trade method
        if trade.direction == Direction.LONG:
            strategy.pos += volume
        else:
            strategy.pos -= volume

        if contract:
            strategy.pos = round_to(strategy.pos, contract.min_volume)

        self.call_strategy_func(strategy, strategy.on_trade, trade)

        # Sync strategy variables to data file
        self.sync_strategy_data(strategy)

        # Update GUI
        self.put_strategy_event(strategy)

    def process_position_event(self, event: Event):
        """"""
        position = event.data

        self.positions.update({position.vt_positionid: position})

        self.offset_converter.update_position(position)

    def check_stop_order(self, tick: TickData):
        """"""
        for stop_order in list(self.stop_orders.values()):
            if stop_order.vt_symbol != tick.vt_symbol:
                continue

            long_triggered = (
                stop_order.direction == Direction.LONG and tick.last_price >= stop_order.price
            )
            short_triggered = (
                stop_order.direction == Direction.SHORT and tick.last_price <= stop_order.price
            )

            if long_triggered or short_triggered:
                strategy = self.strategies[stop_order.strategy_name]

                # To get excuted immediately after stop order is
                # triggered, use limit price if available, otherwise
                # use ask_price_5 or bid_price_5
                if stop_order.direction == Direction.LONG:
                    if tick.limit_up:
                        price = tick.limit_up
                    else:
                        price = tick.ask_price_5
                else:
                    if tick.limit_down:
                        price = tick.limit_down
                    else:
                        price = tick.bid_price_5

                contract = self.main_engine.get_contract(stop_order.vt_symbol)

                vt_orderids = self.send_limit_order(
                    strategy,
                    contract,
                    stop_order.direction,
                    stop_order.offset,
                    price,
                    stop_order.volume,
                    stop_order.lock
                )

                # Update stop order status if placed successfully
                if vt_orderids:
                    # Remove from relation map.
                    self.stop_orders.pop(stop_order.stop_orderid)

                    strategy_vt_orderids = self.strategy_orderid_map[strategy.strategy_name]
                    if stop_order.stop_orderid in strategy_vt_orderids:
                        strategy_vt_orderids.remove(stop_order.stop_orderid)

                    # Change stop order status to cancelled and update to strategy.
                    stop_order.status = StopOrderStatus.TRIGGERED
                    stop_order.vt_orderids = vt_orderids

                    self.call_strategy_func(
                        strategy, strategy.on_stop_order, stop_order
                    )
                    self.put_stop_order_event(stop_order)

    def send_limit_order(
        self,
        strategy: CtaTemplate,
        contract: ContractData,
        direction: Direction,
        offset: Offset,
        price: float,
        volume: float,
        lock: bool
    ):
        """
        Send a limit order to server.
        """
        return self.send_server_order(
            strategy,
            contract,
            direction,
            offset,
            price,
            volume,
            OrderType.LIMIT,
            lock
        )

    def send_server_order(
        self,
        strategy: CtaTemplate,
        contract: ContractData,
        direction: Direction,
        offset: Offset,
        price: float,
        volume: float,
        type: OrderType,
        lock: bool
    ):
        """
        Send a new order to server.
        """
        # Create request and send order.
        original_req = OrderRequest(
            symbol=contract.symbol,
            exchange=contract.exchange,
            direction=direction,
            offset=offset,
            type=type,
            price=price,
            volume=volume,
            reference=f"{APP_NAME}_{strategy.strategy_name}"
        )

        # Convert with offset converter
        req_list = self.offset_converter.convert_order_request(original_req, lock)

        # Send Orders
        vt_orderids = []

        for req in req_list:
            vt_orderid = self.main_engine.send_order(
                req, contract.gateway_name)

            # Check if sending order successful
            if not vt_orderid:
                continue

            vt_orderids.append(vt_orderid)

            self.offset_converter.update_order_request(req, vt_orderid)

            # Save relationship between orderid and strategy.
            self.orderid_strategy_map[vt_orderid] = strategy
            self.strategy_orderid_map[strategy.strategy_name].add(vt_orderid)

        return vt_orderids

    def send_server_stop_order(
        self,
        strategy: CtaTemplate,
        contract: ContractData,
        direction: Direction,
        offset: Offset,
        price: float,
        volume: float,
        lock: bool
    ):
        """
        Send a stop order to server.

        Should only be used if stop order supported
        on the trading server.
        """
        return self.send_server_order(
            strategy,
            contract,
            direction,
            offset,
            price,
            volume,
            OrderType.STOP,
            lock
        )

    def send_local_stop_order(
        self,
        strategy: CtaTemplate,
        direction: Direction,
        offset: Offset,
        price: float,
        volume: float,
        lock: bool
    ):
        """
        Create a new local stop order.
        """
        self.stop_order_count += 1
        stop_orderid = f"{STOPORDER_PREFIX}.{self.stop_order_count}"

        stop_order = StopOrder(
            vt_symbol=strategy.vt_symbol,
            direction=direction,
            offset=offset,
            price=price,
            volume=volume,
            stop_orderid=stop_orderid,
            strategy_name=strategy.strategy_name,
            lock=lock
        )

        self.stop_orders[stop_orderid] = stop_order

        vt_orderids = self.strategy_orderid_map[strategy.strategy_name]
        vt_orderids.add(stop_orderid)

        self.call_strategy_func(strategy, strategy.on_stop_order, stop_order)
        self.put_stop_order_event(stop_order)

        return [stop_orderid]

    @lru_cache()
    def get_data_path(self):
        data_path = os.path.abspath(os.path.join(TRADER_DIR, 'data'))
        return data_path

    def cancel_server_order(self, strategy: CtaTemplate, vt_orderid: str):
        """
        Cancel existing order by vt_orderid.
        """
        order = self.main_engine.get_order(vt_orderid)
        if not order:
            self.write_log(f"撤单失败，找不到委托{vt_orderid}", strategy)
            return

        req = order.create_cancel_request()
        return self.main_engine.cancel_order(req, order.gateway_name)

    def cancel_local_stop_order(self, strategy: CtaTemplate, stop_orderid: str):
        """
        Cancel a local stop order.
        """
        stop_order = self.stop_orders.get(stop_orderid, None)
        if not stop_order:
            return False
        strategy = self.strategies[stop_order.strategy_name]

        # Remove from relation map.
        self.stop_orders.pop(stop_orderid)

        vt_orderids = self.strategy_orderid_map[strategy.strategy_name]
        if stop_orderid in vt_orderids:
            vt_orderids.remove(stop_orderid)

        # Change stop order status to cancelled and update to strategy.
        stop_order.status = StopOrderStatus.CANCELLED

        self.call_strategy_func(strategy, strategy.on_stop_order, stop_order)
        self.put_stop_order_event(stop_order)
        return True

    def send_order(
        self,
        strategy: CtaTemplate,
        direction: Direction,
        offset: Offset,
        price: float,
        volume: float,
        stop: bool,
        lock: bool,
        order_type: OrderType = OrderType.LIMIT,
    ):
        """
        """
        contract = self.main_engine.get_contract(strategy.vt_symbol)
        if not contract:
            self.write_log(f"委托失败，找不到合约：{strategy.vt_symbol}", strategy)
            return ""

        # Round order price and volume to nearest incremental value
        price = round_to(price, contract.pricetick)
        volume = round_to(volume, contract.min_volume)

        if stop:
            if contract.stop_supported:
                return self.send_server_stop_order(strategy, contract, direction, offset, price, volume, lock)
            else:
                return self.send_local_stop_order(strategy, direction, offset, price, volume, lock)
        else:
            return self.send_server_order(strategy, contract, direction, offset, price, volume, type=order_type, lock=lock)

    def cancel_order(self, strategy: CtaTemplate, vt_orderid: str):
        """
        """
        if vt_orderid.startswith(STOPORDER_PREFIX):
            return self.cancel_local_stop_order(strategy, vt_orderid)
        else:
            return self.cancel_server_order(strategy, vt_orderid)

    def cancel_all(self, strategy: CtaTemplate):
        """
        Cancel all active orders of a strategy.
        """
        vt_orderids = self.strategy_orderid_map[strategy.strategy_name]
        if not vt_orderids:
            return

        for vt_orderid in copy(vt_orderids):
            self.cancel_order(strategy, vt_orderid)

    def get_engine_type(self):
        """"""
        return self.engine_type

    def get_pricetick(self, strategy: CtaTemplate):
        """
        Return contract pricetick data.
        """
        contract = self.main_engine.get_contract(strategy.vt_symbol)

        if contract:
            return contract.pricetick
        else:
            return None

    def get_volume_tick(self, strategy: CtaTemplate):
        """
        Return contract min volume data.
        """
        contract = self.main_engine.get_contract(strategy.vt_symbol)

        if contract:
            return contract.min_volume
        else:
            return None

    def get_margin_rate(self, strategy: CtaTemplate):
        """
        返回保证金比率.
        """
        contract = self.main_engine.get_contract(strategy.vt_symbol)

        if contract:
            return contract.margin_rate
        else:
            return None

    def get_size(self, strategy: CtaTemplate):
        """
        返回杠杆倍数.
        """
        contract = self.main_engine.get_contract(strategy.vt_symbol)

        if contract:
            return contract.size
        else:
            return None

    def get_position(self, vt_symbol: str, direction: Direction = Direction.NET, gateway_name: str = ''):
        """
        查询合约在账号的持仓,需要指定方向
        """
        contract = self.main_engine.get_contract(vt_symbol)
        if contract:
            if contract.gateway_name and not gateway_name:
                gateway_name = contract.gateway_name

        vt_position_id = f"{gateway_name}.{vt_symbol}.{direction.value}"
        return self.main_engine.get_position(vt_position_id)

    def get_position_detail(self, vt_symbol):    
        """   
        查询long_pos,short_pos(持仓)，long_pnl,short_pnl(盈亏),active_order(未成交字典)      
        收到PositionHolding类数据     
        """        
        try:        
            return self.offset_converter.get_position_holding(vt_symbol)     
        except:            
            self.write_log(f"当前获取持仓信息为：{self.offset_converter.get_position_holding(vt_symbol)},等待获取持仓信息") 
            position_detail = OrderedDict()
            position_detail.active_orders = {}
            position_detail.long_pos = 0
            position_detail.long_pnl = 0
            position_detail.long_yd = 0
            position_detail.long_td = 0
            position_detail.long_pos_frozen = 0
            position_detail.long_price = 0
            position_detail.short_pos = 0
            position_detail.short_pnl = 0
            position_detail.short_yd = 0
            position_detail.short_td = 0
            position_detail.short_price = 0
            position_detail.short_pos_frozen = 0
            return position_detail

    def load_bar(
        self,
        vt_symbol: str,
        days: int,
        interval: Interval,
        callback: Callable[[BarData], None],
        use_database: bool
    ):
        """"""
        symbol, exchange = extract_vt_symbol(vt_symbol)
        end = datetime.now(get_localzone())
        end = datetime(year=end.year, month=end.month, day=end.day, hour=end.hour, minute=0)
        start = end - timedelta(days)
        bars = []

        # Pass gateway and RQData if use_database set to True
        if not use_database:
            # Query bars from gateway if available
            contract = self.main_engine.get_contract(vt_symbol)

            if contract and contract.history_data:
                req = HistoryRequest(
                    symbol=symbol,
                    exchange=exchange,
                    interval=interval,
                    start=start,
                    end=end
                )
                bars = self.main_engine.query_history(req, contract.gateway_name)

            # Try to query bars from RQData, if not found, load from database.
            else:
                bars = self.query_bar_from_rq(symbol, exchange, interval, start, end)

        if not bars:
            bars = database_manager.load_bar_data(
                symbol=symbol,
                exchange=exchange,
                interval=interval,
                start=start,
                end=end,
            )

        for bar in bars:
            callback(bar)

    def load_tick(
        self,
        vt_symbol: str,
        days: int,
        callback: Callable[[TickData], None]
    ):
        """"""
        symbol, exchange = extract_vt_symbol(vt_symbol)
        end = datetime.now()
        start = end - timedelta(days)

        ticks = database_manager.load_tick_data(
            symbol=symbol,
            exchange=exchange,
            start=start,
            end=end,
        )

        for tick in ticks:
            callback(tick)

    def call_strategy_func(
        self, strategy: CtaTemplate, func: Callable, params: Any = None
    ):
        """
        Call function of a strategy and catch any exception raised.
        """
        try:
            if params:
                func(params)
            else:
                func()
        except Exception:
            strategy.trading = False
            strategy.inited = False

            msg = f"触发异常已停止\n{traceback.format_exc()}"
            self.write_log(msg, strategy)

    def add_strategy(
        self, class_name: str, strategy_name: str, vt_symbol: str, setting: dict
    ):
        """
        Add a new strategy.
        """
        if strategy_name in self.strategies:
            self.write_log(f"创建策略失败，存在重名{strategy_name}")
            return

        strategy_class = self.classes.get(class_name, None)
        if not strategy_class:
            self.write_log(f"创建策略失败，找不到策略类{class_name}")
            return

        strategy = strategy_class(self, strategy_name, vt_symbol, setting)
        self.strategies[strategy_name] = strategy

        # Add vt_symbol to strategy map.
        strategies = self.symbol_strategy_map[vt_symbol]
        strategies.append(strategy)

        # Update to setting file.
        self.update_strategy_setting(strategy_name, setting)

        self.put_strategy_event(strategy)

    def init_strategy(self, strategy_name: str, auto_start: bool = False):
        """
        Init a strategy.
        """
        self.init_executor.submit(self._init_strategy, strategy_name, auto_start=auto_start)

    def _init_strategy(self, strategy_name: str, auto_start: bool = False):
        """
        Init strategies in queue.
        """
        strategy = self.strategies[strategy_name]

        if strategy.inited:
            self.write_log(f"{strategy_name}已经完成初始化，禁止重复操作")
            return

        self.write_log(f"{strategy_name}开始执行初始化")

        # Call on_init function of strategy
        self.call_strategy_func(strategy, strategy.on_init)

        # Restore strategy data(variables)
        data = self.strategy_data.get(strategy_name, None)
        if data:
            for name in strategy.variables:
                value = data.get(name, None)
                if value:
                    setattr(strategy, name, value)

        # Subscribe market data
        contract = self.main_engine.get_contract(strategy.vt_symbol)
        if contract:
            req = SubscribeRequest(
                symbol=contract.symbol, exchange=contract.exchange)
            self.main_engine.subscribe(req, contract.gateway_name)
        else:
            self.write_log(f"行情订阅失败，找不到合约{strategy.vt_symbol}", strategy)

        # Put event to update init completed status.
        strategy.inited = True
        self.put_strategy_event(strategy)
        self.write_log(f"{strategy_name}初始化完成")

        if auto_start:
            self.start_strategy(strategy_name)

    def start_strategy(self, strategy_name: str):
        """
        Start a strategy.
        """
        strategy = self.strategies[strategy_name]
        if not strategy.inited:
            self.write_log(f"策略{strategy.strategy_name}启动失败，请先初始化")
            return

        if strategy.trading:
            self.write_log(f"{strategy_name}已经启动，请勿重复操作")
            return

        self.call_strategy_func(strategy, strategy.on_start)
        strategy.trading = True

        self.put_strategy_event(strategy)

    def stop_strategy(self, strategy_name: str):
        """
        Stop a strategy.
        """
        strategy = self.strategies[strategy_name]
        if not strategy.trading:
            return

        # Call on_stop function of the strategy
        self.call_strategy_func(strategy, strategy.on_stop)

        # Change trading status of strategy to False
        strategy.trading = False

        # Cancel all orders of the strategy
        self.cancel_all(strategy)

        # Sync strategy variables to data file
        self.sync_strategy_data(strategy)

        # Update GUI
        self.put_strategy_event(strategy)

    def edit_strategy(self, strategy_name: str, setting: dict):
        """
        Edit parameters of a strategy.
        """
        strategy = self.strategies[strategy_name]
        strategy.update_setting(setting)

        self.update_strategy_setting(strategy_name, setting)
        self.put_strategy_event(strategy)

    def remove_strategy(self, strategy_name: str):
        """
        Remove a strategy.
        """
        strategy = self.strategies[strategy_name]
        if strategy.trading:
            self.write_log(f"策略{strategy.strategy_name}移除失败，请先停止")
            return

        # Remove setting
        self.remove_strategy_setting(strategy_name)

        # Remove from symbol strategy map
        strategies = self.symbol_strategy_map[strategy.vt_symbol]
        strategies.remove(strategy)

        # Remove from active orderid map
        if strategy_name in self.strategy_orderid_map:
            vt_orderids = self.strategy_orderid_map.pop(strategy_name)

            # Remove vt_orderid strategy map
            for vt_orderid in vt_orderids:
                if vt_orderid in self.orderid_strategy_map:
                    self.orderid_strategy_map.pop(vt_orderid)

        # Remove from strategies
        self.strategies.pop(strategy_name)

        return True

    def reload_strategy(self, strategy_name: str, vt_symbol: str = '', setting: dict = {}):
        """
        重新加载策略
        一般使用于在线更新策略代码，或者更新策略参数，需要重新启动策略
        """
        self.write_log(f'开始重新加载策略{strategy_name}')
        
        # 优先判断重启的策略，是否已经加载
        if strategy_name not in self.strategies or strategy_name not in self.strategy_setting:
            err_msg = f"{strategy_name}不在运行策略中，不能重启"
            self.write_error(err_msg)
            return False, err_msg

        # 从本地配置文件中读取
        if len(setting) == 0:
            strategies_setting = load_json(self.setting_filename)
            old_strategy_config = strategies_setting.get(strategy_name, {})
        else:
            old_strategy_config = copy(self.strategy_setting[strategy_name])

        class_name = old_strategy_config.get('class_name')
        if len(vt_symbol) == 0:
            vt_symbol = old_strategy_config.get('vt_symbol')
        if len(setting) == 0:
            setting = old_strategy_config.get('setting')

        # 停止当前策略实例的运行，撤单
        self.stop_strategy(strategy_name)

        # 移除运行中的策略实例
        self.remove_strategy(strategy_name)

        # 重新添加策略
        self.add_strategy(class_name=class_name,
                          strategy_name=strategy_name,
                          vt_symbol=vt_symbol,
                          setting=setting,
                          auto_init=old_strategy_config.get('auto_init', True),
                          auto_start=old_strategy_config.get('auto_start', True))

        msg = f'成功重载策略{strategy_name}'
        self.write_log(msg)
        return True, msg

    def load_strategy_class(self):
        """
        Load strategy class from source code.
        """
        path1 = Path(__file__).parent.joinpath("strategies")
        self.load_strategy_class_from_folder(
            path1, "vnpy.app.cta_strategy.strategies")

        path2 = Path.cwd().joinpath("strategies")
        self.load_strategy_class_from_folder(path2, "strategies")

    def load_strategy_class_from_folder(self, path: Path, module_name: str = ""):
        """
        Load strategy class from certain folder.
        """
        for dirpath, dirnames, filenames in os.walk(str(path)):
            for filename in filenames:
                if filename.split(".")[-1] in ("py", "pyd", "so"):
                    strategy_module_name = ".".join([module_name, filename.split(".")[0]])
                    self.load_strategy_class_from_module(strategy_module_name)

    def load_strategy_class_from_module(self, module_name: str):
        """
        Load strategy class from module file.
        """
        try:
            module = importlib.import_module(module_name)

            for name in dir(module):
                value = getattr(module, name)
                if (isinstance(value, type) and issubclass(value, CtaTemplate) and value is not CtaTemplate):
                    self.classes[value.__name__] = value
        except:  # noqa
            msg = f"策略文件{module_name}加载失败，触发异常：\n{traceback.format_exc()}"
            self.write_log(msg)

    def load_strategy_data(self):
        """
        Load strategy data from json file.
        """
        self.strategy_data = load_json(self.data_filename)

    def sync_strategy_data(self, strategy: CtaTemplate):
        """
        Sync strategy data into json file.
        """
        data = strategy.get_variables()
        data.pop("inited")      # Strategy status (inited, trading) should not be synced.
        data.pop("trading")

        self.strategy_data[strategy.strategy_name] = data
        save_json(self.data_filename, self.strategy_data)

    def get_all_strategy_class_names(self):
        """
        Return names of strategy classes loaded.
        """
        return list(self.classes.keys())

    def get_strategy_status(self):
        """
        return strategy inited/trading status
        """
        return {k: {'inited': v.inited, 'trading': v.trading} for k, v in self.strategies.items()}

    def get_strategy_pos(self, name, strategy=None):
        """
        获取策略的持仓字典
        :param name:策略名
        :return: [ {},{}]
        """
        # 兼容处理，如果strategy是None，通过name获取
        if strategy is None:
            if name not in self.strategies:
                self.write_log(u'getStategyPos 策略实例不存在：' + name)
                return []
            # 获取策略实例
            strategy = self.strategies[name]

        pos_list = []

        if strategy.inited:
            # 如果策略具有getPositions得方法，则调用该方法
            if hasattr(strategy, 'get_positions'):
                pos_list = strategy.get_positions()
                for pos in pos_list:
                    vt_symbol = pos.get('vt_symbol', None)
                    if vt_symbol:
                        symbol, exchange = extract_vt_symbol(vt_symbol)
                        pos.update({'symbol': symbol})

            # 如果策略有 ctaPosition属性
            elif hasattr(strategy, 'position') and issubclass(strategy.position, CtaPosition):
                symbol, exchange = extract_vt_symbol(strategy.vt_symbol)
                # 多仓
                long_pos = {}
                long_pos['vt_symbol'] = strategy.vt_symbol
                long_pos['symbol'] = symbol
                long_pos['direction'] = 'long'
                long_pos['volume'] = strategy.position.long_pos
                if long_pos['volume'] > 0:
                    pos_list.append(long_pos)

                # 空仓
                short_pos = {}
                short_pos['vt_symbol'] = strategy.vt_symbol
                short_pos['symbol'] = symbol
                short_pos['direction'] = 'short'
                short_pos['volume'] = abs(strategy.position.short_pos)
                if short_pos['volume'] > 0:
                    pos_list.append(short_pos)

            # 获取模板缺省pos属性
            elif hasattr(strategy, 'pos') and isinstance(strategy.pos, int):
                symbol, exchange = extract_vt_symbol(strategy.vt_symbol)
                if strategy.pos > 0:
                    long_pos = {}
                    long_pos['vt_symbol'] = strategy.vt_symbol
                    long_pos['symbol'] = symbol
                    long_pos['direction'] = 'long'
                    long_pos['volume'] = strategy.pos
                    if long_pos['volume'] > 0:
                        pos_list.append(long_pos)
                elif strategy.pos < 0:
                    short_pos = {}
                    short_pos['symbol'] = symbol
                    short_pos['vt_symbol'] = strategy.vt_symbol
                    short_pos['direction'] = 'short'
                    short_pos['volume'] = abs(strategy.pos)
                    if short_pos['volume'] > 0:
                        pos_list.append(short_pos)

        # update local pos dict
        self.strategy_pos_dict.update({name: pos_list})

        return pos_list

    def get_all_strategy_pos(self):
        """
        获取所有得策略仓位明细
        """
        strategy_pos_list = []
        for strategy_name in list(self.strategies.keys()):
            d = OrderedDict()
            d['accountid'] = self.engine_config.get('accountid', '-')
            d['strategy_group'] = self.engine_config.get('strategy_group', self.engine_name)
            d['strategy_name'] = strategy_name
            dt = datetime.now()
            d['trading_day'] = dt.strftime('%Y-%m-%d')
            d['datetime'] = datetime.now()
            strategy = self.strategies.get(strategy_name)
            d['inited'] = strategy.inited
            d['trading'] = strategy.trading
            try:
                d['pos'] = self.get_strategy_pos(name=strategy_name)
            except Exception as ex:
                self.write_error(
                    u'get_strategy_pos exception:{},{}'.format(str(ex), traceback.format_exc()))
                d['pos'] = []
            strategy_pos_list.append(d)

        return strategy_pos_list

    def get_strategy_class_parameters(self, class_name: str):
        """
        Get default parameters of a strategy class.
        """
        strategy_class = self.classes[class_name]

        parameters = {}
        for name in strategy_class.parameters:
            parameters[name] = getattr(strategy_class, name)

        return parameters

    def get_strategy_parameters(self, strategy_name):
        """
        Get parameters of a strategy.
        """
        strategy = self.strategies[strategy_name]
        return strategy.get_parameters()

    def init_all_strategies(self, auto_start: bool = False):
        """
        """
        for strategy_name in self.strategies.keys():
            self.init_strategy(strategy_name, auto_start=auto_start)

    def start_all_strategies(self):
        """
        """
        for strategy_name in self.strategies.keys():
            self.start_strategy(strategy_name)

    def stop_all_strategies(self):
        """
        """
        for strategy_name in self.strategies.keys():
            self.stop_strategy(strategy_name)

    def load_strategy_setting(self):
        """
        Load setting file.
        """
        # 读取引擎得配置
        self.engine_config = load_json(self.engine_filename)

        self.strategy_setting = load_json(self.setting_filename)

        for strategy_name, strategy_config in self.strategy_setting.items():
            self.add_strategy(
                strategy_config["class_name"],
                strategy_name,
                strategy_config["vt_symbol"],
                strategy_config["setting"]
            )

    def update_strategy_setting(self, strategy_name: str, setting: dict):
        """
        Update setting file.
        """
        strategy = self.strategies[strategy_name]

        self.strategy_setting[strategy_name] = {
            "class_name": strategy.__class__.__name__,
            "vt_symbol": strategy.vt_symbol,
            "setting": setting,
        }
        save_json(self.setting_filename, self.strategy_setting)

    def remove_strategy_setting(self, strategy_name: str):
        """
        Update setting file.
        """
        if strategy_name not in self.strategy_setting:
            return

        self.strategy_setting.pop(strategy_name)
        save_json(self.setting_filename, self.strategy_setting)

    def put_stop_order_event(self, stop_order: StopOrder):
        """
        Put an event to update stop order status.
        """
        event = Event(EVENT_CTA_STOPORDER, stop_order)
        self.event_engine.put(event)

    def put_strategy_event(self, strategy: CtaTemplate):
        """
        Put an event to update strategy status.
        """
        data = strategy.get_data()
        event = Event(EVENT_CTA_STRATEGY, data)
        self.event_engine.put(event)

    def write_log(self, msg: str, strategy: CtaTemplate = None, level: int = logging.INFO):
        """
        Create cta engine log event.
        """
        if strategy:
            msg = f"{strategy.strategy_name}: {msg}"

        if strategy:
            strategy_logger = self.strategy_loggers.get(strategy.strategy_name, None)
            if not strategy_logger:
                log_path = get_folder_path('log')
                log_filename = str(log_path.joinpath(str(strategy.strategy_name)))
                print(u'create logger:{}'.format(log_filename))
                self.strategy_loggers[strategy.strategy_name] = setup_logger(file_name=log_filename,
                                                                    name=str(strategy.strategy_name))
                strategy_logger = self.strategy_loggers.get(strategy.strategy_name)
            if strategy_logger:
                strategy_logger.log(level, msg)

        # 如果日志数据异常，错误和告警，输出至sys.stderr
        if level in [logging.CRITICAL, logging.ERROR, logging.WARNING]:
            print(f"{strategy.strategy_name}: {msg}" if strategy.strategy_name else msg, file=sys.stderr)

        msg = msg.replace("\033[1;32;31m", "")
        msg = msg.replace("\033[0m", "")
        log = LogData(msg=msg, gateway_name=APP_NAME)
        event = Event(type=EVENT_CTA_LOG, data=log)
        self.event_engine.put(event)

    def write_error(self, msg: str, strategy: CtaTemplate = None):
        """
        写入错误日志
        """
        self.write_log(msg=msg, strategy=strategy, level=logging.ERROR)
        
    def send_email(self, msg: str, strategy: CtaTemplate = None):
        """
        Send email to default receiver.
        """
        if strategy:
            subject = f"{strategy.strategy_name}"
        else:
            subject = "CTA策略引擎"

        self.main_engine.send_email(subject, msg)

    def get_none_strategy_pos_list(self):
        """获取非策略持有的仓位"""
        # 格式 [  'strategy_name':'account', 'pos': [{'vt_symbol': '', 'direction': 'xxx', 'volume':xxx }] } ]
        none_strategy_pos_file = os.path.abspath(os.path.join(os.getcwd(), 'data', 'none_strategy_pos.json'))
        if not os.path.exists(none_strategy_pos_file):
            return []
        try:
            with open(none_strategy_pos_file, encoding='utf8') as f:
                pos_list = json.load(f)
                if isinstance(pos_list, list):
                    return pos_list

            return []
        except Exception as ex:
            self.write_error(u'未能读取或解释{}'.format(none_strategy_pos_file))
            return []

    def compare_pos(self, strategy_pos_list=[], auto_balance=False):
        """
        对比账号&策略的持仓,不同的话则发出邮件提醒
        """
        # 当前没有接入网关
        if len(self.main_engine.gateways) == 0:
            return False, u'当前没有接入网关'

        self.write_log(u'开始对比账号&策略的持仓')

        # 获取当前策略得持仓
        if len(strategy_pos_list) == 0:
            strategy_pos_list = self.get_all_strategy_pos()
        self.write_log(u'策略持仓清单:{}'.format(strategy_pos_list))

        none_strategy_pos = self.get_none_strategy_pos_list()
        if len(none_strategy_pos) > 0:
            strategy_pos_list.extend(none_strategy_pos)

        # 需要进行对比得合约集合（来自策略持仓/账号持仓）
        vt_symbols = set()

        # 账号的持仓处理 => compare_pos
        compare_pos = dict()  # vt_symbol: {'账号多单': xx, '账号空单':xxx, '策略空单':[], '策略多单':[]}

        for position in list(self.positions.values()):
            # gateway_name.symbol.exchange => symbol.exchange
            vt_symbol = position.vt_symbol
            vt_symbols.add(vt_symbol)

            compare_pos[vt_symbol] = OrderedDict(
                {
                    "账号净仓": position.volume,
                    '策略空单': 0,
                    '策略多单': 0,
                    '空单策略': [],
                    '多单策略': []
                }
            )

        # 逐一根据策略仓位，与Account_pos进行处理比对
        for strategy_pos in strategy_pos_list:
            for pos in strategy_pos.get('pos', []):
                vt_symbol = pos.get('vt_symbol')
                if not vt_symbol:
                    continue
                vt_symbols.add(vt_symbol)
                symbol_pos = compare_pos.get(vt_symbol, None)
                if symbol_pos is None:
                    self.write_log(u'账号持仓信息获取不到{}，创建一个'.format(vt_symbol))
                    symbol_pos = OrderedDict(
                        {
                            "账号净仓": 0,
                            '策略空单': 0,
                            '策略多单': 0,
                            '空单策略': [],
                            '多单策略': []
                        }
                    )

                if pos.get('direction') == 'short':
                    symbol_pos.update({'策略空单': round(symbol_pos.get('策略空单', 0) + abs(pos.get('volume', 0)), 7)})
                    symbol_pos['空单策略'].append(
                        u'{}({})'.format(strategy_pos['strategy_name'], abs(pos.get('volume', 0))))
                    self.write_log(u'更新{}策略持空仓=>{}'.format(vt_symbol, symbol_pos.get('策略空单', 0)))
                if pos.get('direction') == 'long':
                    symbol_pos.update({'策略多单': round(symbol_pos.get('策略多单', 0) + abs(pos.get('volume', 0)), 7)})
                    symbol_pos['多单策略'].append(
                        u'{}({})'.format(strategy_pos['strategy_name'], abs(pos.get('volume', 0))))
                    self.write_log(u'更新{}策略持多仓=>{}'.format(vt_symbol, symbol_pos.get('策略多单', 0)))

        pos_compare_result = ''
        # 精简输出
        compare_info = ''

        for vt_symbol in sorted(vt_symbols):
            # 发送不一致得结果
            symbol_pos = compare_pos.pop(vt_symbol, None)
            if not symbol_pos:
                self.write_error(f'持仓对比中，找不到{vt_symbol}')
                continue
            net_symbol_pos = round(round(symbol_pos['策略多单'], 7) - round(symbol_pos['策略空单'], 7), 7)

            # 多空都一致
            if round(symbol_pos['账号净仓'], 7) == net_symbol_pos:
                msg = u'{}多空都一致.{}\n'.format(vt_symbol, json.dumps(symbol_pos, indent=2, ensure_ascii=False))
                self.write_log(msg)
                compare_info += msg
            else:
                pos_compare_result += '\n{}: {}'.format(vt_symbol, json.dumps(symbol_pos, indent=2, ensure_ascii=False))
                self.write_error(u'{}不一致:{}'.format(vt_symbol, json.dumps(symbol_pos, indent=2, ensure_ascii=False)))
                compare_info += u'{}不一致:{}\n'.format(vt_symbol, json.dumps(symbol_pos, indent=2, ensure_ascii=False))

                diff_volume = round(symbol_pos['账号净仓'], 7) - net_symbol_pos
                # 账号仓位> 策略仓位, sell
                if diff_volume > 0 and auto_balance:
                    contract = self.main_engine.get_contract(vt_symbol)
                    req = OrderRequest(
                        symbol=contract.symbol,
                        exchange=contract.exchange,
                        direction=Direction.SHORT,
                        offset=Offset.CLOSE,
                        type=OrderType.MARKET,
                        price=0,
                        volume=round(diff_volume,7)
                    )
                    self.write_log(f'卖出{vt_symbol} {req.volume}，平衡仓位')
                    self.main_engine.send_order(req, contract.gateway_name)

                # 账号仓位 < 策略仓位 ,buy
                elif diff_volume < 0 and auto_balance:
                    contract = self.main_engine.get_contract(vt_symbol)
                    req = OrderRequest(
                        symbol=contract.symbol,
                        exchange=contract.exchange,
                        direction=Direction.LONG,
                        offset=Offset.OPEN,
                        type=OrderType.MARKET,
                        price=0,
                        volume=round(-diff_volume, 7)
                    )
                    self.write_log(f'买入{vt_symbol} {req.volume}，平衡仓位')
                    self.main_engine.send_order(req, contract.gateway_name)

        # 不匹配，输入到stdErr通道
        if pos_compare_result != '':
            msg = u'账户{}持仓不匹配: {}' \
                .format(self.engine_config.get('accountid', '-'),
                        pos_compare_result)
            self.send_email(msg)
            ret_msg = u'持仓不匹配: {}' \
                .format(pos_compare_result)
            self.write_error(ret_msg)
            return True, compare_info + ret_msg
        else:
            self.write_log(u'账户持仓与策略一致')
            return True, compare_info
