import sys
from itertools import chain

import numpy as np
from keras_preprocessing import sequence
from Tokenisation import tokenizationText
from Training import fitnessIndexTerm, invertIndex, loadFromCorpus, buildWords
from tensorflowUpdate.ChainCRF import ChainCRF
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Bidirectional, Dropout
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.optimizers import RMSprop
from sklearn.metrics import classification_report

class __5_lbl:
    index2label = {0: 'E', 1: 'S', 2: 'B', 3: 'M', 4: 'WB'}
    label2index = {'E': '0', 'S': 1, 'B': '2', 'M': 3, 'WB': 4}

    nbLabels = 5
    np.random.seed(1443)  # for reproducibility

    model_ca = 'data/CA/modelWeight_5seg_ca.hdf5'
    #model_msa = 'data/MSA/modelWeight_5seg_msa.hdf5'
    model_msa = model_ca

    tokens = 0
    case_seg = set()

    def __init__(self, type_model, setting_model):
        self.setting_model = setting_model
        self.index2word = None
        self.word2index = None
        self.type_model = type_model
        self.history = None  # load history of train
        self.train_path = "data/" + type_model + "/train"
        self.dev_path = "data/" + type_model + "/dev"
        self.test_path = "data/" + type_model + "/test"
        self.x_y_Train, self.x_y_Dev = self.PrepareData()
        self.model = self.buildModel()
        if self.type_model == 'MSA':
            self.model.load_weights(self.model_msa)
        elif self.type_model == 'CA':
            self.model.load_weights(self.model_ca)
        else:
            print('error unknown model')
            sys.exit(0)

    def segment_sentence(self, sen_input):
        x_input = []
        if len(sen_input) < 2:
            print("length sentence must be high to 2")
            return
        w_sen = tokenizationText(sen_input)
        for w in w_sen:
            x_input.append([ch for ch in w])
            x_input[-1].append('WB')

        charResult = np.array([[self.word2index.get(w, self.word2index['<UNK>']) for w in words] for words in x_input],
                              dtype=object)
        charResult = sequence.pad_sequences(charResult, maxlen=self.setting_model['max_len'],
                                            padding='post')  # return the index of chars in array of maxLen

        labelPredict = self.model.predict(charResult, batch_size=64, verbose=False).argmax(-1)[charResult != 0]

        output = []
        lb = 0
        for tok in x_input:
            c = ''
            w = []
            for ch, l in zip(tok, labelPredict[lb:lb + len(tok)]):
                if self.index2label[l] != 'WB':
                    c += ch
                    if self.index2label[l] in ['S', 'E']:
                        w.append(c)
                        c = ''
            output.append('+'.join(w))
            lb += len(tok)

        return ' '.join(output)

    def segment_input(self, input_text):
        seg_sen = []
        for sen in input_text.strip().split('\n'):
            seg_sen.append(self.segment_sentence(sen))
        return '\n'.join(seg_sen)


    def PrepareData(self):
        word_train, lbl_train, _ = loadFromCorpus(self.train_path)
        word_dev, lbl_dev, _ = loadFromCorpus(self.dev_path)
        self.index2word = fitnessIndexTerm(word_train + word_dev, reserved=['<PAD>', '<UNK>'])
        self.word2index = invertIndex(self.index2word)

        # train
        x_train = np.array([[self.word2index[w] for w in words] for words in word_train], dtype=object)
        y_train = np.array([[self.label2index[t] for t in s_tags] for s_tags in lbl_train], dtype=object)
        # dev
        x_dev = np.array([[self.word2index[w] for w in words] for words in word_dev], dtype=object)
        y_dev = np.array([[self.label2index[t] for t in s_tags] for s_tags in lbl_dev], dtype=object)

        seqLen = [len(x_train[i]) for i in range(len(x_train))]
        self.setting_model['max_len'] = max(seqLen)

        return (x_train, y_train), (x_dev, y_dev)

    def buildModel(self):
        rnn_ = Sequential()
        rnn_.add(Embedding(len(self.index2word), self.setting_model['embed_dim'],
                           input_length=self.setting_model['max_len'], name='word_emb', mask_zero=True))
        rnn_.add(Dropout(0.5))
        rnn_.add(Bidirectional(LSTM(self.setting_model['lstm_dim'], return_sequences=True)))
        rnn_.add(Dropout(0.5))
        rnn_.add(TimeDistributed(Dense(self.nbLabels)))
        crf = ChainCRF()
        rnn_.add(crf)
        rnn_.compile(loss=crf.sparse_loss,
                     optimizer=RMSprop(learning_rate=self.setting_model['learn_rate']),
                     metrics=['sparse_categorical_accuracy'])
        return rnn_

    def evaluate(self):

        wordsTest, y_test, _ = loadFromCorpus(self.test_path)

        x_test = np.array(
            [[(self.word2index[char] if char in self.word2index else self.word2index['<UNK>']) for char in words]
             for words in wordsTest],
            dtype=object)
        x_test = sequence.pad_sequences(x_test, maxlen=self.setting_model['max_len'], padding='post')

        label_test_pred = self.model.predict(x_test, batch_size=64).argmax(-1)[x_test > 0]

        srcs = list(chain.from_iterable(words for words in wordsTest))
        y_preds = [self.index2label[x] for x in label_test_pred]
        y_refs = list(chain.from_iterable(tags for tags in y_test))
        print("Evaluation Char-based segmentation:")
        char_based = classification_report(
            y_refs,
            y_preds,
            digits=4,
            target_names=list(set(y_refs + y_preds))
        )
        print(char_based)

        _, rtags, ptags = buildWords(srcs, y_refs, y_preds)
        print("Evaluation Word-based segmentation:")
        word_based = '\n'.join(classification_report(
            rtags,
            ptags,
            digits=4,
            target_names=list(set(rtags + ptags)),
            zero_division=1
        ).split('\n')[-4:])
        print(word_based)
        return  char_based, word_based

if __name__ == "__main__":
    type_mode = 'CA'
    __setting = {
        'max_len': None,
        'embed_dim': 200,
        'lstm_dim': 200,
        'learn_rate': 0.01
    }
    model = __5_lbl(type_mode, __setting)
    #print(model.segment_sentence('نسأل الله السلامة والعافية'))
    model.evaluate()