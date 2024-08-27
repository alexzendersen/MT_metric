from utils.metrics import *
import pandas as pd


class Metrics(object):
    def __init__(self, sources: list[str],  machine_translations: list[str], references: list[str]):
        self.metrics_values = {'iou': [], 'ter': [], 'meteor': [], 'bleu': [], 'bert': [], 'comet': []}
        assert len(machine_translations) == len(references), "Wrong data, check the count of MTs and ref`s"
        self.texts = [(source, mt, ref) for source, mt, ref in zip(sources, machine_translations, references)]

    def calculate(self,
                  meteor_gamma:float = 0.5,
                  meteor_beta: float = 3) -> None:
        
        iou_computer = IOU_Metric()
        ter_computer = TER_Metric()
        meteor_computer = Meteor_Metric()
        bleu_computer = BLEU_Metric()
        bert_computer = BertScore_Metric()
        comet_computer = CometMetric(huggingface_token='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')

        for source, mt, ref in self.texts:
            self.metrics_values['iou'].append(iou_computer.score(mt, ref))
            self.metrics_values['ter'].append(ter_computer.score(mt, ref))
            self.metrics_values['meteor'].append(meteor_computer.score(mt, ref, meteor_gamma, meteor_beta))
            self.metrics_values['bleu'].append(bleu_computer.score(mt, ref))
            self.metrics_values['bert'].append(bert_computer.score(mt, ref))
            self.metrics_values['comet'].append(comet_computer.call([source], [mt], [ref])[0])
            
    def to_csv(self, filename: str = "metrics"):
        dt = pd.DataFrame(self.metrics_values)
        dt.to_csv(filename + '.csv')