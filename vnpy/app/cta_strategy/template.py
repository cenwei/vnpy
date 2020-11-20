""""""
from abc import ABC
from copy import copy
import datetime
import os
import sys
import traceback
from typing import Any, Callable
import logging
from vnpy.component.cta_grid_trade import CtaGrid, CtaGridTrade
from vnpy.component.cta_position import CtaPosition

from vnpy.trader.constant import Interval, Direction, Offset, OrderType, Status
from vnpy.trader.object import BarData, TickData, OrderData, TradeData
from vnpy.trader.utility import virtual, append_data

from .base import StopOrder, EngineType


class CtaTemplate(ABC):
    """"""

    author = ""
    parameters = []
    variables = []

    def __init__(
        self,
        cta_engine: Any,
        strategy_name: str,
        vt_symbol: str,
        setting: dict,
    ):
        """"""
        self.cta_engine = cta_engine
        self.strategy_name = strategy_name
        self.vt_symbol = vt_symbol

        self.inited = False
        self.trading = False
        self.pos = 0
        self.entrust = 0  # 是否正在委托, 0, 无委托 , 1, 委托方向是LONG， -1, 委托方向是SHORT

        # Copy a new variables list here to avoid duplicate insert when multiple
        # strategy instances are created with the same strategy class.
        self.variables = copy(self.variables)
        self.variables.insert(0, "inited")
        self.variables.insert(1, "trading")
        self.variables.insert(2, "pos")
        self.variables.insert(3, "entrust")

        self.update_setting(setting)

    def update_setting(self, setting: dict):
        """
        Update strategy parameter wtih value in setting dict.
        """
        for name in self.parameters:
            if name in setting:
                setattr(self, name, setting[name])

    @classmethod
    def get_class_parameters(cls):
        """
        Get default parameters dict of strategy class.
        """
        class_parameters = {}
        for name in cls.parameters:
            class_parameters[name] = getattr(cls, name)
        return class_parameters

    def get_parameters(self):
        """
        Get strategy parameters dict.
        """
        strategy_parameters = {}
        for name in self.parameters:
            strategy_parameters[name] = getattr(self, name)
        return strategy_parameters

    def get_variables(self):
        """
        Get strategy variables dict.
        """
        strategy_variables = {}
        for name in self.variables:
            strategy_variables[name] = getattr(self, name)
        return strategy_variables

    def get_data(self):
        """
        Get strategy data.
        """
        strategy_data = {
            "strategy_name": self.strategy_name,
            "vt_symbol": self.vt_symbol,
            "class_name": self.__class__.__name__,
            "author": self.author,
            "parameters": self.get_parameters(),
            "variables": self.get_variables(),
        }
        return strategy_data

    @virtual
    def on_init(self):
        """
        Callback when strategy is inited.
        """
        pass

    @virtual
    def on_start(self):
        """
        Callback when strategy is started.
        """
        pass

    @virtual
    def on_stop(self):
        """
        Callback when strategy is stopped.
        """
        pass

    @virtual
    def on_tick(self, tick: TickData):
        """
        Callback of new tick data update.
        """
        pass

    @virtual
    def on_bar(self, bar: BarData):
        """
        Callback of new bar data update.
        """
        pass

    @virtual
    def on_trade(self, trade: TradeData):
        """
        Callback of new trade data update.
        """
        pass

    @virtual
    def on_order(self, order: OrderData):
        """
        Callback of new order data update.
        """
        pass

    @virtual
    def on_stop_order(self, stop_order: StopOrder):
        """
        Callback of stop order update.
        """
        pass

    def buy(self, price: float, volume: float, stop: bool = False, lock: bool = False,
            vt_symbol: str = '', order_type: OrderType = OrderType.LIMIT, order_time: datetime = None, grid: CtaGrid = None):
        """
        Send buy order to open a long position.
        """
        return self.send_order(vt_symbol, Direction.LONG, Offset.OPEN, price, volume, stop, lock, order_type=order_type, order_time=order_time, grid=grid)

    def sell(self, price: float, volume: float, stop: bool = False, lock: bool = False,
            vt_symbol: str = '', order_type: OrderType = OrderType.LIMIT, order_time: datetime = None, grid: CtaGrid = None):
        """
        Send sell order to close a long position.
        """
        return self.send_order(vt_symbol, Direction.SHORT, Offset.CLOSE, price, volume, stop, lock, order_type=order_type, order_time=order_time, grid=grid)

    def short(self, price: float, volume: float, stop: bool = False, lock: bool = False,
            vt_symbol: str = '', order_type: OrderType = OrderType.LIMIT, order_time: datetime = None, grid: CtaGrid = None):
        """
        Send short order to open as short position.
        """
        return self.send_order(vt_symbol, Direction.SHORT, Offset.OPEN, price, volume, stop, lock, order_type=order_type, order_time=order_time, grid=grid)

    def cover(self, price: float, volume: float, stop: bool = False, lock: bool = False,
            vt_symbol: str = '', order_type: OrderType = OrderType.LIMIT, order_time: datetime = None, grid: CtaGrid = None):
        """
        Send cover order to close a short position.
        """
        return self.send_order(vt_symbol, Direction.LONG, Offset.CLOSE, price, volume, stop, lock, order_type=order_type, order_time=order_time, grid=grid)

    def send_order(
        self,
        vt_symbol: str,
        direction: Direction,
        offset: Offset,
        price: float,
        volume: float,
        stop: bool = False,
        lock: bool = False,
        order_type: OrderType = OrderType.LIMIT,
        order_time: datetime = None,
        grid: CtaGrid = None
    ):
        """
        Send a new order.
        """

        if vt_symbol == '':
            vt_symbol = self.vt_symbol

        if self.trading:
            if direction == Direction.LONG:
                self.entrust = 1
            elif direction == Direction.SHORT:
                self.entrust = -1
            
            vt_orderids = self.cta_engine.send_order(
                self, direction, offset, price, volume, stop, lock
            )
            for vt_orderid in vt_orderids:
                d = {
                    'direction': direction,
                    'offset': offset,
                    'vt_symbol': vt_symbol,
                    'price': price,
                    'volume': volume,
                    'order_type': order_type,
                    'traded': 0,
                    'order_time': order_time,
                    'status': Status.SUBMITTING
                }
                if grid:
                    d.update({'grid': grid})
                    grid.order_ids.append(vt_orderid)

            return vt_orderids
        else:
            return []

    def cancel_order(self, vt_orderid: str):
        """
        Cancel an existing order.
        """
        if self.trading:
            self.cta_engine.cancel_order(self, vt_orderid)

    def cancel_all(self):
        """
        Cancel all orders sent by strategy.
        """
        if self.trading:
            self.entrust = 0
            self.cta_engine.cancel_all(self)

    def write_log(self, msg: str, level: int = logging.INFO):
        """
        Write a log message.
        """
        self.cta_engine.write_log(msg, self, level)

    def get_engine_type(self):
        """
        Return whether the cta_engine is backtesting or live trading.
        """
        return self.cta_engine.get_engine_type()

    def get_pricetick(self):
        """
        Return pricetick data of trading contract.
        """
        return self.cta_engine.get_pricetick(self)

    def load_bar(
        self,
        days: int,
        interval: Interval = Interval.MINUTE,
        callback: Callable = None,
        use_database: bool = False
    ):
        """
        Load historical bar data for initializing strategy.
        """
        if not callback:
            callback = self.on_bar

        self.cta_engine.load_bar(
            self.vt_symbol,
            days,
            interval,
            callback,
            use_database
        )

    def load_tick(self, days: int):
        """
        Load historical tick data for initializing strategy.
        """
        self.cta_engine.load_tick(self.vt_symbol, days, self.on_tick)

    def put_event(self):
        """
        Put an strategy data event for ui update.
        """
        if self.inited:
            self.cta_engine.put_strategy_event(self)

    def send_email(self, msg):
        """
        Send email to default receiver.
        """
        if self.inited:
            self.cta_engine.send_email(msg, self)

    def sync_data(self):
        """
        Sync strategy variables value into disk storage.
        """
        if self.trading:
            self.cta_engine.sync_strategy_data(self)


class CtaSignal(ABC):
    """"""

    def __init__(self):
        """"""
        self.signal_pos = 0

    @virtual
    def on_tick(self, tick: TickData):
        """
        Callback of new tick data update.
        """
        pass

    @virtual
    def on_bar(self, bar: BarData):
        """
        Callback of new bar data update.
        """
        pass

    def set_signal_pos(self, pos):
        """"""
        self.signal_pos = pos

    def get_signal_pos(self):
        """"""
        return self.signal_pos


class TargetPosTemplate(CtaTemplate):
    """"""
    tick_add = 1

    last_tick = None
    last_bar = None
    target_pos = 0

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        """"""
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)

        self.active_orderids = []
        self.cancel_orderids = []

        self.variables.append("target_pos")

    @virtual
    def on_tick(self, tick: TickData):
        """
        Callback of new tick data update.
        """
        self.last_tick = tick

        if self.trading:
            self.trade()

    @virtual
    def on_bar(self, bar: BarData):
        """
        Callback of new bar data update.
        """
        self.last_bar = bar

    @virtual
    def on_order(self, order: OrderData):
        """
        Callback of new order data update.
        """
        vt_orderid = order.vt_orderid

        if not order.is_active():
            if vt_orderid in self.active_orderids:
                self.active_orderids.remove(vt_orderid)

            if vt_orderid in self.cancel_orderids:
                self.cancel_orderids.remove(vt_orderid)

    def check_order_finished(self):
        """"""
        if self.active_orderids:
            return False
        else:
            return True

    def set_target_pos(self, target_pos):
        """"""
        self.target_pos = target_pos
        self.trade()

    def trade(self):
        """"""
        if not self.check_order_finished():
            self.cancel_old_order()
        else:
            self.send_new_order()

    def cancel_old_order(self):
        """"""
        for vt_orderid in self.active_orderids:
            if vt_orderid not in self.cancel_orderids:
                self.cancel_order(vt_orderid)
                self.cancel_orderids.append(vt_orderid)

    def send_new_order(self):
        """"""
        pos_change = self.target_pos - self.pos
        if not pos_change:
            return

        long_price = 0
        short_price = 0

        if self.last_tick:
            if pos_change > 0:
                long_price = self.last_tick.ask_price_1 + self.tick_add
                if self.last_tick.limit_up:
                    long_price = min(long_price, self.last_tick.limit_up)
            else:
                short_price = self.last_tick.bid_price_1 - self.tick_add
                if self.last_tick.limit_down:
                    short_price = max(short_price, self.last_tick.limit_down)

        else:
            if pos_change > 0:
                long_price = self.last_bar.close_price + self.tick_add
            else:
                short_price = self.last_bar.close_price - self.tick_add

        if self.get_engine_type() == EngineType.BACKTESTING:
            if pos_change > 0:
                vt_orderids = self.buy(long_price, abs(pos_change))
            else:
                vt_orderids = self.short(short_price, abs(pos_change))
            self.active_orderids.extend(vt_orderids)

        else:
            if self.active_orderids:
                return

            if pos_change > 0:
                if self.pos < 0:
                    if pos_change < abs(self.pos):
                        vt_orderids = self.cover(long_price, pos_change)
                    else:
                        vt_orderids = self.cover(long_price, abs(self.pos))
                else:
                    vt_orderids = self.buy(long_price, abs(pos_change))
            else:
                if self.pos > 0:
                    if abs(pos_change) < self.pos:
                        vt_orderids = self.sell(short_price, abs(pos_change))
                    else:
                        vt_orderids = self.sell(short_price, abs(self.pos))
                else:
                    vt_orderids = self.short(short_price, abs(pos_change))
            self.active_orderids.extend(vt_orderids)

class CryptoFutureTemplate(CtaTemplate):
    """
    数字货币合约期货模板
    """

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        self.position:CtaPosition = None  # 仓位组件
        self.gt:CtaGridTrade = None  # 网格交易组件

        super().__init__(
            cta_engine, strategy_name, vt_symbol, setting
        )

        # 增加仓位管理模块
        self.position = CtaPosition(strategy=self)
        self.position.maxPos = sys.maxsize
        # 增加网格持久化模块
        self.gt = CtaGridTrade(strategy=self)

    def init_position(self):
        """
        初始化Positin
        使用网格的持久化，获取开仓状态的多空单，更新
        :return:
        """
        self.write_log(u'init_position(),初始化持仓')
        changed = False
        if len(self.gt.up_grids) <= 0:
            self.position.short_pos = 0
            # 加载已开仓的空单数据，网格JSON
            short_grids = self.gt.load(direction=Direction.SHORT, open_status_filter=[True])
            if len(short_grids) == 0:
                self.write_log(u'没有持久化的空单数据')
                self.gt.up_grids = []

            else:
                self.gt.up_grids = short_grids
                for sg in short_grids:
                    if len(sg.order_ids) > 0 or sg.order_status:
                        self.write_log(f'重置委托状态:{sg.order_status},清除委托单：{sg.order_ids}')
                        sg.order_status = False
                        [self.cancel_order(vt_orderid) for vt_orderid in sg.order_ids]
                        sg.order_ids = []
                        changed = True

                    self.write_log(u'加载持仓空单[{},价格:{},数量:{}手,开仓时间:{}'
                                   .format(self.vt_symbol, sg.open_price,
                                           sg.volume, sg.open_time))
                    self.position.short_pos = round(self.position.short_pos - sg.volume, 7)

                self.write_log(u'持久化空单，共持仓:{}手'.format(abs(self.position.short_pos)))

        if len(self.gt.dn_grids) <= 0:
            # 加载已开仓的多数据，网格JSON
            self.position.long_pos = 0
            long_grids = self.gt.load(direction=Direction.LONG, open_status_filter=[True])
            if len(long_grids) == 0:
                self.write_log(u'没有持久化的多单数据')
                self.gt.dn_grids = []
            else:
                self.gt.dn_grids = long_grids
                for lg in long_grids:

                    if len(lg.order_ids) > 0 or lg.order_status:
                        self.write_log(f'重置委托状态:{lg.order_status},清除委托单：{lg.order_ids}')
                        lg.order_status = False
                        [self.cancel_order(vt_orderid) for vt_orderid in lg.order_ids]
                        lg.order_ids = []
                        changed = True

                    self.write_log(u'加载持仓多单[{},价格:{},数量:{}手, 开仓时间:{}'
                                   .format(self.vt_symbol, lg.open_price, lg.volume, lg.open_time))
                    self.position.long_pos = round(self.position.long_pos + lg.volume, 7)

                self.write_log(f'持久化多单，共持仓:{self.position.long_pos}手')

        self.position.pos = round(self.position.long_pos + self.position.short_pos, 7)

        self.write_log(u'{}加载持久化数据完成，多单:{}，空单:{},共:{}手'
                       .format(self.strategy_name,
                               self.position.long_pos,
                               abs(self.position.short_pos),
                               self.position.pos))
        self.pos = self.position.pos
        if changed:
            self.gt.save()
        self.display_grids()

    def display_grids(self):
        """更新网格显示信息"""
        if not self.inited:
            return
        self.account_pos = self.cta_engine.get_position(vt_symbol=self.vt_symbol, direction=Direction.NET)
        if self.account_pos:
            self.write_log(
                f'账号{self.vt_symbol}持仓:{self.account_pos.volume}, 冻结:{self.account_pos.frozen}, 盈亏:{self.account_pos.pnl}')

        up_grids_info = ""
        for grid in list(self.gt.up_grids):
            if not grid.open_status and grid.order_status:
                up_grids_info += f'平空中: [已平:{grid.traded_volume} => 目标:{grid.volume}, 委托时间:{grid.order_time}]\n'
                if len(grid.order_ids) > 0:
                    up_grids_info += f'委托单号:{grid.order_ids}'
                continue

            if grid.open_status and not grid.order_status:
                up_grids_info += f'持空中: [数量:{grid.volume}, 开仓时间:{grid.open_time}]\n'
                continue

            if not grid.open_status and grid.order_status:
                up_grids_info += f'开空中: [已开:{grid.traded_volume} => 目标:{grid.volume}, 委托时间:{grid.order_time}]\n'
                if len(grid.order_ids) > 0:
                    up_grids_info += f'委托单号:{grid.order_ids}'

        dn_grids_info = ""
        for grid in list(self.gt.dn_grids):
            if not grid.open_status and grid.order_status:
                dn_grids_info += f'平多中: [已平:{grid.traded_volume} => 目标:{grid.volume}, 委托时间:{grid.order_time}]\n'
                if len(grid.order_ids) > 0:
                    dn_grids_info += f'委托单号:{grid.order_ids}'
                continue

            if grid.open_status and not grid.order_status:
                dn_grids_info += f'持多中: [数量:{grid.volume}\n, 开仓时间:{grid.open_time}]\n'
                continue

            if not grid.open_status and grid.order_status:
                dn_grids_info += f'开多中: [已开:{grid.traded_volume} => 目标:{grid.volume}, 委托时间:{grid.order_time}]\n'
                if len(grid.order_ids) > 0:
                    dn_grids_info += f'委托单号:{grid.order_ids}'

        if len(up_grids_info) > 0:
            self.write_log(up_grids_info)
        if len(dn_grids_info) > 0:
            self.write_log(dn_grids_info)

    def on_trade(self, trade: TradeData):
        """交易更新"""
        self.write_log(u'{},交易更新:{},当前持仓：{} '
                       .format(self.cur_datetime,
                               trade.__dict__,
                               self.position.pos))

        dist_record = dict()
        if self.backtesting:
            dist_record['datetime'] = trade.time
        else:
            dist_record['datetime'] = ' '.join([self.cur_datetime.strftime('%Y-%m-%d'), trade.time])
        dist_record['volume'] = trade.volume
        dist_record['price'] = trade.price
        dist_record['margin'] = trade.price * trade.volume * self.cta_engine.get_margin_rate(trade.vt_symbol)
        dist_record['symbol'] = trade.vt_symbol

        if trade.direction == Direction.LONG and trade.offset == Offset.OPEN:
            dist_record['operation'] = 'buy'
            self.position.open_pos(trade.direction, volume=trade.volume)
            dist_record['long_pos'] = self.position.long_pos
            dist_record['short_pos'] = self.position.short_pos

        if trade.direction == Direction.SHORT and trade.offset == Offset.OPEN:
            dist_record['operation'] = 'short'
            self.position.open_pos(trade.direction, volume=trade.volume)
            dist_record['long_pos'] = self.position.long_pos
            dist_record['short_pos'] = self.position.short_pos

        if trade.direction == Direction.LONG and trade.offset != Offset.OPEN:
            dist_record['operation'] = 'cover'
            self.position.close_pos(trade.direction, volume=trade.volume)
            dist_record['long_pos'] = self.position.long_pos
            dist_record['short_pos'] = self.position.short_pos

        if trade.direction == Direction.SHORT and trade.offset != Offset.OPEN:
            dist_record['operation'] = 'sell'
            self.position.close_pos(trade.direction, volume=trade.volume)
            dist_record['long_pos'] = self.position.long_pos
            dist_record['short_pos'] = self.position.short_pos

        self.save_dist(dist_record)
        self.pos = self.position.pos

    def save_dist(self, dist_data):
        """
        保存策略逻辑过程记录=》 csv文件按
        :param dist_data:
        :return:
        """
        if self.backtesting:
            save_path = self.cta_engine.get_logs_path()
        else:
            save_path = self.cta_engine.get_data_path()
        try:
            if 'margin' not in dist_data:
                dist_data.update({'margin': dist_data.get('price', 0) * dist_data.get('volume',
                                                                                      0) * self.cta_engine.get_margin_rate(
                    dist_data.get('symbol', self.vt_symbol))})
            if 'datetime' not in dist_data:
                dist_data.update({'datetime': self.cur_datetime})
            if self.position and 'long_pos' not in dist_data:
                dist_data.update({'long_pos': self.position.long_pos})
            if self.position and 'short_pos' not in dist_data:
                dist_data.update({'short_pos': self.position.short_pos})

            file_name = os.path.abspath(os.path.join(save_path, f'{self.strategy_name}_dist.csv'))
            append_data(file_name=file_name, dict_data=dist_data, field_names=self.dist_fieldnames)
        except Exception as ex:
            self.write_error(u'save_dist 异常:{} {}'.format(str(ex), traceback.format_exc()))
