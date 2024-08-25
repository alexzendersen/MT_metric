import re
from argparse import ArgumentParser
import pandas as pd
from utils.metrics import *
from utils.test_tools import *
import class_metric 

#зедсь будут метрики, из которых будет соcтавлять веткор. В последнем столбце будуд значения новой метрики 
# metrics_data = {
      
# }


def valid_check(filename: str) -> bool:
    return len(re.findall('\w+.csv$', filename)) != 0

if __name__ == "__main__":
        parser = ArgumentParser()

        parser.add_argument("--data", help="CSV datafile", type=str, default="")

        config = parser.parse_args()

        assert valid_check(config.data), "Incorrect name of data file"

        filename = 'data/' + config.data

        data = pd.read_csv(filename)

        refs = list(data['reference'])
        mts = list(data['machine_translation'])
        sources = list(data['source'])

        metrics = class_metric.Metrics(sources, mts, refs)

        metrics.calculate()
        metrics.to_csv()

        # df = pd.DataFrame(metrics_data)
        # # проходимся 
        # for row in data.itertuples():
        #     pass
        



        