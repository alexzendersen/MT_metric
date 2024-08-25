from utils.metrics import *
import pandas as pd


class Metrics(object):
    def __init__(self, sources: list[str],  machine_translations: list[str], references: list[str]):
        self.metrics_values = {'iou': [], 'ter': [], 'meteor': [], 'comet': [], 'prob_metric': []}
        assert len(machine_translations) == len(references), "Wrong data, check the count of MTs and ref`s"
        self.texts = [(source, mt, ref) for source, mt, ref in zip(sources, machine_translations, references)]

    def calculate(self,
                  meteor_gamma:float = 0.5,
                  meteor_beta: float = 3) -> None:
        for source, mt, ref in self.texts:
            self.metrics_values['ter'].append(calculate_ter(mt, ref))
            self.metrics_values['meteor'].append(calculate_meteor(mt, ref, meteor_gamma, meteor_beta))
            self.metrics_values['iou'].append(0)
            self.metrics_values['prob_metric'].append(0)

            comet = CometMetric(huggingface_token='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')
            self.metrics_values['comet'].append(comet.call([source], [mt], [ref]))

            
    def to_csv(self, filename: str = "metrics"):
        dt = pd.DataFrame(self.metrics_values)
        dt.to_csv(filename + '.csv')