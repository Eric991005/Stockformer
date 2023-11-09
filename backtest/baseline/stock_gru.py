import qlib
import pandas as pd
from qlib.utils.time import Freq
from qlib.contrib.evaluate import backtest_daily
from qlib.contrib.evaluate import risk_analysis
from qlib.contrib.strategy import TopkDropoutStrategy 
from qlib.utils import init_instance_by_config 
from qlib.backtest.high_performance_ds import  NumpyQuote,BaseQuote
from qlib.backtest.exchange import  Exchange

from qlib.workflow import R #
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord #

from qlib.contrib.report import analysis_model, analysis_position

from qlib.backtest import backtest, executor,exchange  

from qlib.backtest import high_performance_ds


if __name__ == '__main__':
    # provider_uri='/root/test/STOCK/hh_f_all_data'
    qlib.init(provider_uri = '~/.qlib/qlib_data/hh_f_all_data')
    MARKET = "all"
    BENCHMARK = "000300"
    EXP_NAME = "tutorial_exp"
    CSI300_BENCH = "000300"

    # 数据参数
    handler_kwargs = {
            "start_time": "2020-01-01",
            "end_time": "2022-12-31",
            "fit_start_time": "2020-01-01",
            "fit_end_time": "2022-05-31",
            "instruments": '2',
    }

    # 因子生成参数
    handler_conf = {
        "class": "Alpha158",
        "module_path": "qlib.contrib.data.handler",
        "kwargs": handler_kwargs,
    }

    hd = init_instance_by_config(handler_conf)

    # 数据集参数
    dataset_conf = {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": hd,
                "segments": {
                    "train": ("2020-01-01", "2021-12-31"),
                    "valid": ("2022-01-01", "2022-05-31"),
                    "test": ("2022-06-02", "2022-12-12"),
                },
            },
    }

    dataset = init_instance_by_config(dataset_conf)

    #### 模型训练  LSTM
    model = init_instance_by_config({
            "class": "GRU",
            "module_path": "qlib.contrib.model.pytorch_gru",
            "kwargs": {
                "d_feat": 158,
                },
    })


    # start exp to train model
    with R.start(experiment_name=EXP_NAME):
        model.fit(dataset)
        R.save_objects(trained_model=model)

        rec = R.get_recorder()
        rid = rec.id # save the record id

        # Inference and saving signal
        sr = SignalRecord(model, dataset, rec)
        sr.generate()
    
    # load recorder
    recorder = R.get_recorder(recorder_id=rid, experiment_name=EXP_NAME)
    pred_df = recorder.load_object("pred.pkl")


    FREQ = "day"

    STRATEGY_CONFIG = {
        "topk": 10,
        "n_drop": 2,
        # pred_score, pd.Series
        "signal": pred_df,
    }

    EXECUTOR_CONFIG = {
        "time_per_step": "day",
        "generate_portfolio_metrics": True,
    }

    backtest_config = {
        "start_time": "2022-06-02",
        "end_time": "2022-12-12",
        "account": 100000000,
        "benchmark": '000300',
        "exchange_kwargs": {
            "freq": FREQ,
            "limit_threshold": 0.095,
            "deal_price": "close",
            "open_cost": 0.0005,
            "close_cost": 0.0015,
            "min_cost": 5,
        },
    }


    # strategy object
    strategy_obj = TopkDropoutStrategy(**STRATEGY_CONFIG)
    # executor object
    executor_obj = executor.SimulatorExecutor(**EXECUTOR_CONFIG)
    # backtest
    portfolio_metric_dict, indicator_dict = backtest(executor=executor_obj, strategy=strategy_obj, **backtest_config)
    analysis_freq = "{0}{1}".format(*Freq.parse(FREQ))
    # backtest info
    report_normal, positions_normal = portfolio_metric_dict.get(analysis_freq)

    report_normal.to_csv('gru_return.csv')
    pred_df.to_csv('gru_pred.csv')