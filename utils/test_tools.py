import re
from nltk import ngrams

import deep_translator
from deep_translator import GoogleTranslator

import simplemma
from simplemma import text_lemmatizer



# from metrics.py import calculate_iou
LANGDICT = {'georgian': 'ka', 'english': 'en', 'russian': 'ru'}
    
def translate2lang(text, lang='engilsh'):
    translated_text = GoogleTranslator(source='auto', target=lang).translate(text)
    return translated_text

def preprocess_text(text):
    """Функция для очистки текста."""
    # Приводим текст к нижнему регистру
    text = text.lower()
    # Удаляем знаки препинания, цифры и лишние пробелы
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()  # Заменяем несколько пробелов одним и удаляем начальные и конечные пробелы
    return text

def generate_ngrams(words_list, n):
    """Генерирует всевозможные n-граммы из списка слов."""
    # Очищаем и преобразуем список
    cleaned_words = [preprocess_text(word) for word in words_list]
    # Генерируем n-граммы
    ngrams_list = []
    for i in range(n+1):
        for word in cleaned_words:
            ngrams_list.extend([' '.join(gram) for gram in list(ngrams(word.split(), i))])
    return ngrams_list

def process_texts(reference: str, translation: str, keywords: list, lang='english') -> float:
    foreign_keywords = [translate2lang(keyword, lang=lang) for keyword in keywords]
    
    reference_keywords = set(re.findall(r'\b(' + '|'.join(foreign_keywords) + r')\b', reference.lower()))
    translation_keywords = set(re.findall(r'\b(' + '|'.join(foreign_keywords) + r')\b', translation.lower()))
    
    iou = calculate_iou(reference_keywords, translation_keywords)
    
    return iou

def lemmatize(text, lang='english'):
    av_langs = list(LANGDICT.keys())
    assert lang in av_langs, f'Language {lang} is not availiable now. The following languages are available: {av_langs}'
    language = LANGDICT.get(lang)
    return ' '.join(text_lemmatizer(text, lang=language))