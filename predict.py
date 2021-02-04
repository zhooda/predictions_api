import keras
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

class Predict:
    def __init__(
        self,
        vocabulary_size=10000,
        maxlen=684
    ):
        self.model = load_model('models/predict.h5')
        self.vocabulary_size = vocabulary_size
        self.maxlen = maxlen
        self.tokenizer = Tokenizer(num_words=self.vocabulary_size)
        df = pd.read_csv('models/train.csv', encoding='utf-8')
        self.tokenizer.fit_on_texts(df['text'].astype('str').to_numpy())

    def get_summary(self):
        return self.model.summary()

    def __prepare_text(self, text):
        arr = np.array([text])
        x_test_seq = self.tokenizer.texts_to_sequences(arr)
        x_test_seq_padded = pad_sequences(x_test_seq, maxlen=self.maxlen)
        return x_test_seq_padded
    
    def get_prediction(self, text):
        processed = self.__prepare_text(text)
        y_test_pred = self.model.predict(processed)
        return y_test_pred[0][0]

if __name__ == "__main__":
    p = Predict()
    print(p.get_summary())
    print(p.get_prediction('haha epochs go brrrr'))
    print(p.get_prediction('well uh it definitely is accurate'))
    print(p.get_prediction('hello my love'))