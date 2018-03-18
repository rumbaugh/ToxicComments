import time, sys, os, re, csv, codecs, cPickle, numpy as np, pandas as pd
import xgboost as xgb
from sklearn import decomposition,neighbors
from sklearn.neighbors import KDTree,BallTree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, learning_curve, train_test_split
from sklearn.cross_validation import KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import log_loss, roc_auc_score
import matplotlib.pyplot as plt
import cPickle
from ProcessText import TagText

from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, SpatialDropout1D, concatenate
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.callbacks import Callback

import warnings

gbm_params = {
    #'learning_rate' : 0.02,
 'n_estimators': 2000,
 'max_depth': 4,
 'min_child_weight': 2,
 #'gamma':1,
 'gamma':0.9,                        
 'subsample':0.8,
 'colsample_bytree':0.8,
 'objective': 'binary:logistic',
 'nthread': -1,
 'scale_pos_weight':1}



pd.options.mode.chained_assignment = None

def shuffle(x):
    rinds = np.arange(len(x))
    np.random.shuffle(rinds)
    return x[rinds]

class SklearnHelper(object):
    def __init__(self, clf, seed=None, params=None):
        if seed != None: params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x, prob = False):
        if prob:
            return self.clf.predict_proba(x)[:,1]
        else:
            return self.clf.predict(x)
    
    def fit(self,x,y):
        return self.clf.fit(x,y)
    
    def feature_importances(self,x,y, verbose = True):
        try:
            self.importances = self.clf.feature_importances_
        except AttributeError:
            self.importances = self.clf.fit(x,y).feature_importances_
        if verbose: self.print_importances(x.columns.tolist())
        print(self.importances)

    def print_importances(self, colnames):
        isrt = np.argsort(self.importances)
        for i in isrt: 
            print('{}: {:.6f}'.format(colnames[i], self.importances[i]))

    def Kfold(self, x_train, y_ytrain, NCV = 5):
        return cross_val_score(self.clf, x_train, y_train, cv = NCV)

def get_oof(clf, x_train, y_train, x_test, NFOLDS = 5, prob = False):
    ntrain, ntest = x_train.shape[0], x_test.shape[0]
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))
    shuffle_inds = np.arange(ntrain)
    np.random.shuffle(shuffle_inds)

    for ikf in range(0, NFOLDS):
        test_index, train_index = shuffle_inds[int(ikf*ntrain/NFOLDS):int((ikf+1)*ntrain/NFOLDS)], np.append(shuffle_inds[:int(ikf*ntrain/NFOLDS)], shuffle_inds[int((ikf+1)*ntrain/NFOLDS):])
        x_tr = x_train.iloc[train_index]
        y_tr = y_train.iloc[train_index]
        x_te = x_train.iloc[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te, prob = prob)
        oof_test_skf[ikf, :] = clf.predict(x_test, prob = prob)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

def get_oof_probs(clf, train, test, NFOLDS = 5, outfile = None, params = {'n_estimators': 100}, verbose = False, output_train_probs = False):
    st = time.time()
    X_train, Y, X_test = train.loc[:,'Length':], train.loc[:,'toxic':'identity_hate'], test.loc[:,'Length':]
    outdf = pd.DataFrame({'id': test['id']})
    if output_train_probs: outdf_train = pd.DataFrame({'id': test['id']})
    clfs = {}
    for tox in Y.columns.tolist():
        clfs[tox] = SklearnHelper(clf, params=params)
        oof_train, oof_test = get_oof(clfs[tox], X_train, Y[tox], X_test, prob = True)
        outdf[tox] = oof_test
        if output_train_probs: outdf_train[tox] = oof_train
        if verbose:
            print 'Finished {}. Elapsed time: {:.2f} seconds.'.format(tox, time.time() - st)
    if outfile != None: outdf.to_csv(outfile, index = False)
    if output_train_probs:
        return outdf, clfs, outdf_train
    else:
        return outdf, clfs

def check_bad_probs(probs, train, y = 'toxic', eps = 1e-15, tag = True, start = 0):
    try:
        y = train[y]
    except:
        pass
    #epsprobs = np.copy(probs)
    #epsprobs[epsprobs < eps] = eps
    #epsprobs[1 - epsprobs < eps] = 1 - eps
    #log_losses = -(y*np.log(epsprobs) + (1 - y) * np.log(1 - epsprobs))
    log_losses = np.abs(y - probs)
    bad_inds = np.argsort(log_losses)[len(log_losses)-1::-1].values
    inp = ''
    i = start
    while inp == '':
        clean_str = 'Clean'
        if train.toxic.iloc[bad_inds[i]]: clean_str = 'Toxic'
        for tox in ['severe_toxic', "obscene", "threat", "insult", "identity_hate"]:
            if train[tox].iloc[bad_inds[i]]:
                clean_str = '{},'.format(clean_str)
                for curstr in tox.split('_'): clean_str = '{} {}{}'.format(clean_str, curstr[0].upper(), curstr[1])
        print '\n({}) ID: {} - {}'.format(i, train['id'].iloc[bad_inds[i]], clean_str)
        print 'Probability: {} - {}'.format(y.name, probs[bad_inds[i]])
        print 'Log Loss: {}'.format(log_losses[bad_inds[i]])
        print train.comment_text.iloc[bad_inds[i]]
        print TagText(train.comment_text.iloc[bad_inds[i]])
        inp = raw_input('')
        i += 1

def get_train_probs(clf_dict, train):
    prob_dict = {}
    for tox in clf_dict.keys():
        prob_dict[tox] = clf_dict[tox].predict(train.loc[:,'Length':], prob = True)
    return pd.DataFrame(prob_dict)

def compare_probs(df1, df2, true):
    compare_cols = np.intersect1d(df1.columns.tolist(), df2.columns.tolist())
    assert 'id' in compare_cols, 'id not in dfs'
    assert len(compare_cols) > 1, 'no intersection beyond id'
    df = pd.merge(df1.loc[:,compare_cols], df2.loc[:,compare_cols], on = ['id'], suffixes = ('_1','_2'))
    assert df.shape[1] == 2*len(compare_cols) - 1, 'Size of df is not right: {}, {}'.format(df.shape, len(compare_cols))
    assert df.shape[0] > 0, 'merge on id failed'
    df = pd.merge(df, true, on = ['id'], suffixes = ('_0', ''))
    assert df.shape[0] > 0, 'true merge on id failed'
    better_metric = np.zeros(0)
    for col in compare_cols[compare_cols != 'id']:
        better_metric = np.append(better_metric, np.abs(df['{}_1'.format(col)].values - df[col].values) - np.abs(df['{}_2'.format(col)].values - df[col].values))
    return better_metric
    
                                            
class ToxicFit():
    def __init__(self, train = None, test = None, datadir = '.', testmask = None):
        try:
            if train == None: train = 'processed_wtags.csv'
            self.train = pd.read_csv('{}/{}'.format(datadir, train))
        except:
            self.train = train
        try:
            self.test = pd.read_csv('{}/{}'.format(datadir, test))
        except:
            self.test = test
        if testmask != None:
            testmask = pd.read_csv('internal_test_inds.csv').in_test.values
            self.train, self.test = self.train[testmask == 1], self.train[testmask == 0]
            

    def set_fold(self, ntrain, NFOLDS = 5):
        shuffle_inds = np.arange(ntrain)
        np.random.shuffle(shuffle_inds)
        self.fold = np.zeros(ntrain,dtype = 'i4')
        for ikf in range(1, NFOLDS):
            self.fold[shuffle_inds[int(ikf*ntrain/NFOLDS):int((ikf+1)*ntrain/NFOLDS)]] = ikf
        
    def RF_oof_fit(self, targets = ['toxic', 'severe_toxic', "obscene", "threat", "insult", "identity_hate"], NFOLDS = 5, prob = True, params = {'n_estimators': 100}, clf = RandomForestClassifier, dotest = False, progressive = False, verbose = True):
        st = time.time()
        X_train, Y = self.train.loc[:,'Length':], self.train[targets]
        ntrain = X_train.shape[0]
        self.oof_train = pd.DataFrame(np.zeros((ntrain, len(targets))), columns = targets)
        if dotest: 
            X_test = self.test.loc[:,'Length':]
            ntest = X_test.shape[0]
            self.oof_test = pd.DataFrame(np.zeros((ntest, len(targets))), columns = targets)
        try:
            self.fold
        except AttributeError:
            self.set_fold(ntrain, NFOLDS = NFOLDS)
        self.clfs = pd.DataFrame(np.zeros((NFOLDS, len(targets)), dtype = 'object'), columns = targets)
        for target in targets:
            if dotest: oof_test_skf = np.empty((NFOLDS, ntest))
            for ikf in range(0, NFOLDS):
                x_tr = X_train[self.fold != ikf]
                y_tr = Y[self.fold != ikf]
                x_te = X_train[self.fold == ikf]
                self.clfs[target].iloc[ikf] = SklearnHelper(clf, params = params)
                self.clfs[target].iloc[ikf].train(x_tr, y_tr[target])

                self.oof_train[target][self.fold == ikf] = self.clfs[target].iloc[ikf].predict(x_te, prob = prob)
                if dotest:
                    oof_test_skf[ikf, :] = self.clfs[target].iloc[ikf].predict(X_test, prob = prob)
            if dotest:
                self.oof_test[target] = oof_test_skf.mean(axis=0)
            if progressive:
                X_train['{}_probs'.format(target)] = self.oof_train[target].values
                self.X_train = X_train
                if dotest:
                    X_test['{}_probs'.format(target)] = self.oof_test[target].values
            if verbose:
                print 'Finished {}. Elapsed time: {:.2f} seconds.'.format(target, time.time() - st)

    def keras_oof_fit(self, EMBEDDING_FILE='/home/rumbaugh/Downloads/glove.6B.50d.txt', embed_size = 50, max_features = 20000, maxlen = 100, NFOLDS = 5, prob = True, dotest = True):
        list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
        ntrain = self.train.shape[0]
        self.oof_train = pd.DataFrame(np.zeros((ntrain, len(list_classes))), columns = list_classes)
        if dotest: 
            ntest = self.test.shape[0]
            self.oof_test = pd.DataFrame(np.zeros((ntest, len(list_classes))), columns = list_classes)
            oof_test_skf = np.zeros((ntest, len(list_classes), NFOLDS))
        try:
            self.fold
        except AttributeError:
            self.set_fold(ntrain, NFOLDS = NFOLDS)
        list_sentences_finaltest = self.test["comment_text"].fillna("_na_").values
        for ikf in range(0, NFOLDS):
            train = self.train[self.fold != ikf]
            test = self.train[self.fold == ikf]
            list_sentences_train = train["comment_text"].fillna("_na_").values
            y = train[list_classes].values
            list_sentences_test = test["comment_text"].fillna("_na_").values
            tokenizer = Tokenizer(num_words=max_features)
            tokenizer.fit_on_texts(list(list_sentences_train))
            list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
            list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
            list_tokenized_finaltest = tokenizer.texts_to_sequences(list_sentences_finaltest)
            X_test = pad_sequences(list_tokenized_finaltest, maxlen=maxlen)
            X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
            X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)
            def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
            embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE))
            all_embs = np.stack(embeddings_index.values())
            emb_mean,emb_std = all_embs.mean(), all_embs.std()

            word_index = tokenizer.word_index
            nb_words = min(max_features, len(word_index))
            embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
            for word, i in word_index.items():
                if i >= max_features: continue
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None: embedding_matrix[i] = embedding_vector
            inp = Input(shape=(maxlen,))
            x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
            x = Bidirectional(LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
            x = GlobalMaxPool1D()(x)
            x = Dense(50, activation="relu")(x)
            x = Dropout(0.1)(x)
            x = Dense(6, activation="sigmoid")(x)
            model = Model(inputs=inp, outputs=x)
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            model.fit(X_t, y, batch_size=32, epochs=2, validation_split=0.0)
            y_train = model.predict([X_te], batch_size=1024, verbose=1)
            y_test = model.predict([X_test], batch_size=1024, verbose=1)
            self.oof_train[self.fold == ikf] = y_train
            if dotest: oof_test_skf[:,:,ikf] = y_test
        if dotest: self.oof_test = pd.DataFrame(oof_test_skf.mean(axis=2), columns = list_classes)

    def gru_fit(self, outfile=None, EMBEDDING_FILE='/home/rumbaugh/ToxicComments/crawl-300d-2M.vec', max_features=30000, maxlen=100, embed_size=300, batch_size=32, epochs=2):
        list_sentences_test = self.test["comment_text"].fillna("_na_").values
        list_sentences_train = self.train["comment_text"].fillna("_na_").values
        targets=["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
        tokenizer = text.Tokenizer(num_words=max_features)
        tokenizer.fit_on_texts(list(list_sentences_train) + list(list_sentences_test))
        X_train = tokenizer.texts_to_sequences(list_sentences_train)
        X_test = tokenizer.texts_to_sequences(list_sentences_test)
        x_train = sequence.pad_sequences(X_train, maxlen=maxlen)
        x_test = sequence.pad_sequences(X_test, maxlen=maxlen)
        y_train = self.train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
        def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
        embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE))

        word_index = tokenizer.word_index
        nb_words = min(max_features, len(word_index))
        embedding_matrix = np.zeros((nb_words, embed_size))
        for word, i in word_index.items():
            if i-1 >= max_features: continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None: embedding_matrix[i-1] = embedding_vector

        inp = Input(shape=(maxlen, ))
        x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
        x = SpatialDropout1D(0.2)(x)
        x = Bidirectional(GRU(80, return_sequences=True))(x)
        avg_pool = GlobalAveragePooling1D()(x)
        max_pool = GlobalMaxPooling1D()(x)
        conc = concatenate([avg_pool, max_pool])
        outp = Dense(6, activation="sigmoid")(conc)
    
        model = Model(inputs=inp, outputs=outp)
        model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

        X_tra, X_val, y_tra, y_val = train_test_split(x_train, y_train, train_size=0.95)
        RocAuc = RocAucEvaluation(validation_data=(X_val, y_val), interval=1)

        hist = model.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val),callbacks=[RocAuc], verbose=2)

        y_pred = model.predict(x_test, batch_size=1024)
        self.outdf = pd.DataFrame(np.zeros((self.test.shape[0], len(targets))), columns = targets)
        self.outdf[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_pred
        self.outdf['id'] = self.test['id']
        if outfile is not None:
            self.outdf.to_csv(outfile, index=False)


    def stack_pred(self, clf_files_train, clf_files_test=None, clf=xgb.XGBClassifier, params=gbm_params, targets=["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"], outfile=None, verbose=True, dotest=True):
        # Loads output from previous fits to training and test data and creates a stacked model based on those predictions
        # clf_files_train and clf_files_test are lists of the names of csv files. Each csv file (should) have the columns in targets (plus 'id'). 
        st = time.time()
        if np.shape(clf_files_test) == ():
            clf_files_test = pd.Series(clf_files_train).apply(lambda x: x.replace('train','test')).values
        assert len(clf_files_train) == len(clf_files_test), 'Train and test file arrays are not equal'
        assert np.count_nonzero(clf_files_train == clf_files_test) == 0, 'Train and test files are the same'
        names = np.copy(clf_files_train)
        for i in range(0, len(names)):
            try:
                names[i] = re.findall('oofprobs_[a-zA-Z0-9]+', names[i])[0][9:]
            except IndexError:
                pass
        train_df_dict = {names[x]: pd.read_csv(clf_files_train[x]) for x in range(0,len(names))}
        test_df_dict = {names[x]: pd.read_csv(clf_files_test[x]) for x in  range(0,len(names))}
        assert len(train_df_dict.keys()) == len(test_df_dict.keys()), 'train and test dicts differ in length'
        outdf = pd.DataFrame(np.zeros((self.test.shape[0], len(targets))), columns = targets)
        outdf['id'] = self.test['id']
        self.clfs = {x: None for x in targets}
        test_with_dummy_variables = self.test.copy()
        for target in targets:
            test_with_dummy_variables[target] = 0
        if verbose: print 'Beginning fits...'
        for target in targets:
            train = self.train.copy()
            test = test_with_dummy_variables.copy()
            for name in names:
                df = train_df_dict[name]
                train = pd.merge(train, df, on = ['id'], suffixes = ('', '_{}'.format(name)))
                df = test_df_dict[name]
                test = pd.merge(test, df, on = ['id'], suffixes = ('', '_{}'.format(name)))
            train.drop('id', axis=1, inplace=True)
            test.drop(np.append(['id'], targets), axis=1, inplace=True)

            X_train, y = train.drop(targets, axis = 1), train[target]
            
            if not(dotest): continue
            self.clfs[target] = SklearnHelper(clf, params = params)
            assert np.count_nonzero(np.in1d(X_train.columns.tolist(), test.columns.tolist(), invert = True)) == 0, "Columns in train and test don't match:\n{}\n{}".format(X_train.columns.tolist(), test.columns.tolist())
            self.clfs[target].train(X_train, y)
            outdf[target] = self.clfs[target].predict(test, prob = True)
            if verbose: print 'Finished {}. Elapsed time: {:.2f} seconds'.format(target, time.time() - st)
        if outfile != None: outdf.to_csv(outfile, index = False)

class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch+1, score))
