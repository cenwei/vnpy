from datetime import datetime
import multiprocessing
import sys
import os
from time import sleep
from logging import INFO
import signal

# 将repostory的目录i，作为根目录，添加到系统环境中。
VNPY_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if VNPY_ROOT not in sys.path:
    sys.path.append(VNPY_ROOT)
    print(f'append {VNPY_ROOT} into sys.path')

from vnpy.event import EventEngine
from vnpy.trader.setting import SETTINGS
from vnpy.trader.engine import MainEngine

from vnpy.gateway.binances import BinancesGateway
from vnpy.app.cta_strategy import CtaStrategyApp
from vnpy.app.risk_manager import RiskManagerApp
from vnpy.app.rpc_service import RpcServiceApp
from vnpy.trader.utility import load_json, save_json
from vnpy.app.cta_strategy.base import EVENT_CTA_LOG



def run_child():
    """
    Running in the child process.
    """

    event_engine = EventEngine()
    main_engine = MainEngine(event_engine)
    main_engine.add_gateway(BinancesGateway)
    rpc_engine = main_engine.add_app(RpcServiceApp)
    cta_engine = main_engine.add_app(CtaStrategyApp)
    main_engine.add_app(RiskManagerApp)
    main_engine.write_log("主引擎创建成功")

    log_engine = main_engine.get_engine("log")
    event_engine.register(EVENT_CTA_LOG, log_engine.process_log_event)
    main_engine.write_log("注册日志事件监听")

    binances_setting = load_json("connect_binances.json")
    gateway_name = "BINANCES"

    main_engine.connect(binances_setting, gateway_name)

    main_engine.write_log("连接BINANCES接口")

    sleep(10)
    rpc_setting = load_json("rpc_service_setting.json")
    rpc_engine.start(rpc_setting['rep_address'], rpc_setting['pub_address'])

    cta_engine.init_engine()
    main_engine.write_log("CTA策略初始化完成")

    cta_engine.init_all_strategies(True)
    # sleep(10)   # Leave enough time to complete strategy initialization
    main_engine.write_log("CTA策略全部初始化")
    
    # cta_engine.start_all_strategies()
    # main_engine.write_log("CTA策略全部启动")
    
    def ctrl_handler(signum, frame):
        cta_engine.stop_all_strategies()
        cta_engine.close()
        rpc_engine.close()
        main_engine.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, ctrl_handler)

    while True:
        # connect_data = load_json("proxy_connect_status.json")
        # if not connect_data.get(gateway_name):
        #     main_engine.connect(binances_setting, gateway_name)
        #     connect_data.update(gateway_name, True)
        #     save_json("proxy_connect_status.json", connect_data)
        #     print("重连接口")
        sleep(10)
            
    
        


def run_parent():
    """
    Running in the parent process.
    """
    print("启动CTA策略守护父进程")

    child_process = None
    try:
        while True:

            # Start child process in trading period
            if child_process is None:
                print("启动子进程")
                child_process = multiprocessing.Process(target=run_child)
                child_process.start()
                print("子进程启动成功")

            sleep(5)
    except KeyboardInterrupt:
        print('')


if __name__ == "__main__":
    run_parent()
