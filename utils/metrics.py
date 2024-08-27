import nltk
import torch
import pickle

# import torchmetrics
from torchmetrics.text import TranslationEditRate
from huggingface_hub import login
# import comet
from comet import download_model, load_from_checkpoint
from nltk.translate.bleu_score import sentence_bleu

from evaluate import load

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from string import punctuation


import numpy as np

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

class Metric(object):
  def _str_to_str(sentence: str, language: str)->str:
    lemmatizer = WordNetLemmatizer()
    stem = SnowballStemmer("russian")
    tokens = word_tokenize(sentence)
    if language == "english": #Для английских используется леммантизация
        stop_words = set(stopwords.words(language)).union(set(punctuation))
        filtered_tokens = [word for word in tokens if word not in stop_words]
        stemmed_words = [lemmatizer.lemmatize(word) for word in filtered_tokens]
        return " ".join(stemmed_words)
    elif language == "russian": #Для русских используется стемминг
        stop_words = set(stopwords.words(language)).union(set(punctuation))
        filtered_tokens = [word for word in tokens if word not in stop_words]
        lemmatized_words = [stem.stem(word) for word in filtered_tokens]
        return " ".join(lemmatized_words)
    else:
        raise ValueError("Unknown language")



class IOU_Metric(Metric):
  def __init__(self, language='english'):
    self.language = language
  def score(self, translation : str, reference: str) -> float:
    translation, reference = set(self._str_to_str(translation, self.language)), set(self._str_to_str(reference, self.language))
    intersection = len(translation.intersection(reference))
    union = len(translation.union(reference))
    iou = intersection / union if union != 0 else 0
    return iou
  

class TER_Metric(Metric):
  def __init__(self, language='english'):
    self.language = language
    self.ter = TranslationEditRate()
  def score(self, translation : str, reference: str):
    translation, reference = self._str_to_str(translation, self.language), self._str_to_str(reference, self.language)
    return self.ter(translation, [reference]).item()

class Meteor_Metric(Metric):
  is_downloaded = False
  def __init__(self, language='english'):
    self.language = language
    if not Meteor_Metric.is_downloaded:
      nltk.download('wordnet')
      Meteor_Metric.is_downloaded = True
  def score(self,translation: str, reference: str, gamma: float=0.5, beta: float=3):
    translation, reference = self._str_to_str(translation, self.language), self._str_to_str(reference, self.language)
    results = nltk.translate.meteor_score.meteor_score(hypothesis=str(translation).split(),
                                                              references=[str(reference).split()],
                                                              gamma=gamma,
                                                              beta=beta)
    return results

class BLEU_Metric(Metric):
  def __init__ (self, language='english'):
    self.language = language
  def score(self, translation: str, reference: str)->float:
    translation, reference = self._str_to_str(translation, self.language), self._str_to_str(reference, self.language)
    return sentence_bleu([reference.split()], translation.split())



class BertScore_Metric(Metric):
  def __init__(self, language='english'):
    self.language = language
    self.bertscore = load("bertscore")
  def score(self, translation: str, reference: str)->float:
    translation, reference = self._str_to_str(translation, self.language), self._str_to_str(reference, self.language)
    results = self.bertscore.compute(predictions=[translation], references=[reference], model_type="distilbert-base-uncased")
    return results["f1"][0]

class CometMetric(object):
    def __init__(self, huggingface_token, model_name="Unbabel/XCOMET-XXL", gpus=1 if torch.cuda.is_available() else 0):
        # Login to huggingface
        login(huggingface_token)

        # Choose your model from Hugging Face Hub
        self.model_path_comet = download_model(model_name)
        self.comet = load_from_checkpoint(self.model_path_comet)

        self.gpus = gpus
        
    def _compute_comet(self, sources: list[str], predictions: list[str], references: list[str],
                       batch_size: int = 32) -> list[float]:
        gpus = self.gpus
        data = [{"src": str(src), "mt": str(mt), "ref": str(ref)} for src, mt, ref in zip(sources, predictions, references)]
        results = self.comet.predict(data, batch_size=batch_size, gpus=gpus)
        return results['scores']

    def call(self, sources: list[str], translations: list[str], references: list[str]):
        assert len(sources) == len(translations) and len(translations) == len(references), 'References, translations and sources must be the same length'
        return self._compute_comet(sources, translations, references, batch_size=len(sources))

class WBTranslate(object):
    def init(self, lr_kwargs={}):
        self.lr_kwargs = lr_kwargs
        self.fitted = False
        self.sc = StandardScaler()

    def ready_fit(self, X, y):
        X = self.sc.fit_transform(X)
        self.linear_model = LogisticRegression(**self.lr_kwargs)
        self.linear_model.fit(X, y)
        self.fitted = True
        self.train_score = self.linear_model.score(X, y)
        
    def fit(self, ter_list, comet_list, meteor_list, iou_list, target):
        X = self._construct_dataset(ter_list, comet_list, meteor_list, iou_list)
        self.ready_fit(X, target)

    def ready_compute(self, X):
        assert self.fitted, 'Unfitted error. Call .fit()/.ready_fit() first or specify .fitted=True mannually.'
        X = self.sc.transform(X)
        return self.linear_model.predict_proba(X)[:,1]        

    def compute(self, ter_list, comet_list, meteor_list, iou_list):
        assert self.fitted, 'Unfitted error. Call .fit()/.ready_fit() first or specify .fitted=True mannually.'
        X = self.sc.transform(self._construct_dataset(ter_list, comet_list, meteor_list, iou_list))
        return self.linear_model.predict_proba(X)[:,1]

    def _construct_dataset(self, ter_list, comet_list, meteor_list, iou_list):
        return np.stack([ter_list, comet_list, meteor_list, iou_list], axis=1)

    def load_pickle(self, linear_model_filename: str, scaler_filename: str):
        with open(linear_model_filename, 'rb') as fp:
            linmod = pickle.load(fp)
            
        with open(scaler_filename, 'rb') as fp:
            scaler = pickle.load(fp)

        self.fitted = True
        self.linear_model = linmod
        self.sc = scaler