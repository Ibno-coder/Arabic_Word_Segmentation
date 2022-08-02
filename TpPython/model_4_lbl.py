import sys
from itertools import chain
from time import gmtime, strftime

import numpy as np
from keras_preprocessing import sequence
from tensorflowUpdate.ChainCRF import ChainCRF
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Bidirectional, Dropout
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.optimizers import RMSprop
from Tokenisation import splitText_withSen, loadLblRewerite
from Training import loadFromCorpus_4Seg, fitnessIndexTerm, invertIndex, buildWords_4Seg
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint

class __4_lbl:
    index2label = {0: 'E', 1: 'S', 2: 'B', 3: 'M'}
    label2index = {'E': '0', 'S': 1, 'B': '2', 'M': 3}

    nbLabels = 4
    np.random.seed(1443)  # for reproducibility

    model_ca = 'data/CA/modelWeight_4seg_ca.hdf5'
    new_model_hdf5 = 'data/CA/_4seg_ca.hdf5'  # TODO       # put a new name for model to not forget the old
    new_model_json = 'data/CA/_4seg_ca.json'  # TODO       # put a new name for model to not forget the old
    # model_msa = 'data/MSA/modelWeight_5seg_msa.hdf5'
    model_msa = model_ca

    dictLabel_2 = ('له', 'ليل')
    dictLabel_3 = loadLblRewerite('files/RewriterWords/label_5_cl.txt')
    dictLabel_4 = loadLblRewerite('files/RewriterWords/label_4_cl.txt')

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

    def trainModel(self):

        X_train = sequence.pad_sequences(self.x_y_Train[0], maxlen=self.setting_model['max_len'], padding='post')
        y_train = sequence.pad_sequences(self.x_y_Train[1], maxlen=self.setting_model['max_len'], padding='post')
        y_train = np.expand_dims(y_train, -1)

        X_dev = sequence.pad_sequences(self.x_y_Dev[0], maxlen=self.setting_model['max_len'], padding='post')
        y_dev = sequence.pad_sequences(self.x_y_Dev[1], maxlen=self.setting_model['max_len'], padding='post')
        y_dev = np.expand_dims(y_dev, -1)

        print(strftime("%Y-%m-%d %H:%M:%S", gmtime()) + ' Build model...')


        early_stopping = EarlyStopping(patience=10, verbose=1)
        checkpointer = ModelCheckpoint( self.new_model_hdf5, verbose=1, save_best_only=True)

        model_json = self.model.to_json()

        with open(self.new_model_json, 'w') as json_file:
            json_file.write(model_json)
        print("saved json")
        self.model.fit(x=X_train, y=y_train,
                  validation_data=(X_dev, y_dev),
                  verbose=1,
                  batch_size=64,
                  epochs=self.setting_model['epochs'],
                  callbacks=[early_stopping, checkpointer])
        print(strftime("%Y-%m-%d %H:%M:%S", gmtime()) + ' Save the trained model...')


    def segment_sentence(self, sen_input):
        sen_input = splitText_withSen(sen_input)  # we need just the default words : the text origin
        charResult = np.array(
            [[self.word2index.get(w, self.word2index['<UNK>']) for w in words] for words in sen_input],
            dtype=object)
        charResult = sequence.pad_sequences(charResult, maxlen=self.setting_model['max_len'],
                                            padding='post')  # return the index of chars in array of maxLen

        labelPredict = self.model.predict(charResult, batch_size=64, verbose=False).argmax(-1)[charResult != 0]

        output = []
        lb = 0
        for word in sen_input:
            c = ''
            w = []
            for ch, l in zip(word, labelPredict[lb:lb + len(word)]):
                c += ch
                if self.index2label[l] in ['S', 'E']:
                    if c in self.dictLabel_4:  # label 4
                        c = c[:-1] + 'ى'
                    elif len(c) >= 2 and c.endswith('آ'):  # label 5
                        c = c[:-1] + 'اى'
                    elif len(c) >= 2 and c.startswith('ئ'):  # label 6
                        c = 'إ' + c[1:]
                    w.append(c)
                    c = ''
            if 'ل' in w:
                start = w.index('ل') + 1
                if 'ل' == w[start]:  # label 1
                    w[start] = 'ال'
                elif w[start].startswith('ل'):  # label 2 with the three conflict
                    new_input = 'ا' + str(w[start])
                    if w[start].startswith('لا') or w[start] in self.dictLabel_2:
                        new_input = 'ال' + str(w[start])
                    new_format = self.sub__segment(new_input)
                    w.remove(w[start])
                    for elt in new_format.split('+'):
                        w.insert(start, elt)
                        start += 1
            elif len(w) >= 2 and w[-2].endswith('ت') and w[-2] in self.dictLabel_3:  # label 3
                w[-2] = w[-2][:-1] + 'ة'
            elif len(w) > 2 and w[-2] == 'و':
                w[-2] = 'وا'
            self.tokens += 1
            self.case_seg.add('+'.join(w))
            output.append('+'.join(w))
            lb += len(word)

        return ' '.join(output)

    def sub__segment(self, inputWord):
        inputWord = splitText_withSen(inputWord)
        charResult = np.array(
            [[self.word2index.get(w, self.word2index['<UNK>']) for w in words] for words in inputWord],
            dtype=object)
        charResult = sequence.pad_sequences(charResult, maxlen=self.setting_model['max_len'],
                                            padding='post')  # return the index of chars in array of maxLen

        labelPredict = self.model.predict(charResult, batch_size=64).argmax(-1)[
            charResult != 0]  # return the prediction labels of chars

        output = []
        lb = 0
        for tok in inputWord:
            c = ''
            w = []
            for ch, l in zip(tok, labelPredict[lb:lb + len(tok)]):
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

    def segment_file(self, path_file):
        return

    def PrepareData(self):
        word_train, lbl_train, _ = loadFromCorpus_4Seg(self.train_path)
        word_dev, lbl_dev, _ = loadFromCorpus_4Seg(self.dev_path)
        self.index2word = fitnessIndexTerm(word_train + word_dev, reserved=['<PAD>', '<UNK>'])
        self.word2index = invertIndex(self.index2word)

        # train
        x_train = np.array([[self.word2index[w] for w in words] for words in word_train], dtype=object)
        y_train = np.array([[self.label2index[t] for t in s_tags] for s_tags in lbl_train], dtype=object)
        # dev
        x_dev = np.array([[self.word2index[w] for w in words] for words in word_dev], dtype=object)
        y_dev = np.array([[self.label2index[t] for t in s_tags] for s_tags in lbl_dev], dtype=object)

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


        wordsTest, y_test, _ = loadFromCorpus_4Seg(self.test_path)

        x_test = np.array(
            [[(self.word2index[char] if char in self.word2index else self.word2index['<UNK>']) for char in words]
             for words in wordsTest],
            dtype=object)
        x_test = sequence.pad_sequences(x_test, maxlen=self.setting_model['max_len'], padding='post')

        label_test_pred = self.model.predict(x_test, batch_size=64).argmax(-1)[x_test > 0]

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

        _, rtags, ptags = buildWords_4Seg(wordsTest, y_test, y_preds)
        print("Evaluation Word-based segmentation:")
        word_based = '\n'.join(classification_report(
            rtags,
            ptags,
            digits=4,
            target_names=list(set(rtags + ptags)),
            zero_division=1
        ).split('\n')[-4:])
        print(word_based)
        return char_based, word_based


if __name__ == "__main__":
    type_mode = 'CA'
    ca_setting = {
        'max_len': 80,
        'embed_dim': 200,
        'lstm_dim': 200,
        'learn_rate': 0.01,
        'epochs': 50
    }
    model = __4_lbl(type_mode, ca_setting)
    #print(model.segment_sentence('نسأل الله السلامة والعافية'))
    model.trainModel()