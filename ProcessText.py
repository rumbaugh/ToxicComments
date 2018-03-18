import time, sys, os, re, csv, codecs, cPickle, numpy as np, pandas as pd
import nltk

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers



#remember to make overwrite options and to reuse columns to make ProcessText faster

important_words = ['admin', 'ass', 'bitch', 'block', 'comm', 'crap', 'cunt', 'dick', 'dumb', 'fag', 'fuck', 'gay', 'hate', 'hell', 'hole', 'idiot', 'moron', 'shit', 'slut', 'stupid', 'suck', 'tard', 'tit', 'troll_words', 'very', 'you']

bad_words = ['asperg', 'ass', 'autis', 'ball', 'bastard', 'bitch', 'bloody', 'brat', 'buffoon', 'bull', 'butt', 'cancer', 'chink', 'cock', 'crap', 'cunt', 'damn', 'dick', 'dirty', 'disgust', 'dope', 'douche', 'dumb', 'fag', 'fat', 'filthy', 'fool', 'freak', 'fuck', 'garbage', 'gay', 'geek', 'god', 'gook', 'hell', 'hole', 'homo', 'idiot', 'islam', 'jerk', 'kill', 'loser', 'mental', 'midget', 'moron', 'muslim', 'nasty', 'nerd', 'nig', 'penis', 'phile', 'piss', 'poo', 'prick', 'prostit', 'puss', 'queer', 'scum', 'sex', 'shit', 'shut', 'sick', 'sissy', 'slut', 'stfu', 'stupid', 'suck', 'tard', 'tit', 'trash', 'twat', 'ugl', 'vomit', 'wank', 'wtf']

bigot_words = ['bigot', 'black', 'chink', 'fascis', 'femin', 'gay', 'gook', 'kraut', 'islam', 'jew', 'muslim', 'nazi', 'nig', 'queer', 'racis', 'semit', 'white']

forum_action_words = ['admin','ban','block','delet','spam', 'terminat', 'vandal' ]

debate_words = ['bias', 'coward', 'disgrac', 'disgust', 'dishonest', 'grow up', 'hypocrit', 'ignoran', 'illitera', 'imbecil', 'immature', 'incompeten', 'insult', 'liar', 'lie', 'lyin', 'nonsense', 'pathetic', 'ridiculous', 'rude', 'silly', 'sociopath', 'vile']

other_words = ['comm', 'correct']#, 'very', 'you']

troll_words = ['bab', 'cry', 'death', 'fail', 'fat', 'girl', 'hate', 'joke', 'life', 'lol', 'loser', 'nerd', 'newb', 'noob', 'pathetic', 'rofl', 'salt','sock', 'troll', 'ugl', 'uppet']

toxic_words = np.unique(np.concatenate((bad_words, bigot_words, forum_action_words, troll_words, other_words)))

isolated_words = ['ass', 'ban']

def IPAddress(string):
    if re.match('.*\d+\.\d+\.\d+[\s\'\"]*.*', string, re.DOTALL):
        return 1
    else:
        return 0

def WordFrequencies(df, mostcommon = 100, unique = True):
    clean_text = ' '.join(df[df.loc[:,'toxic':'identity_hate'].sum(axis=1) == 0].comment_text)
    CleanFreqs = nltk.FreqDist(nltk.word_tokenize(clean_text.lower().decode('ascii','ignore').encode('ascii')))
    CleanFreqWords = zip(*CleanFreqs.most_common(mostcommon))[0]
    bad_freqs = {}
    bad_freq_words = {}
    for tox in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']:
        all_text = ' '.join(df[df[tox] == 1].comment_text)
        bad_freqs[tox] = nltk.FreqDist(nltk.word_tokenize(all_text.lower().decode('ascii','ignore').encode('ascii')))
        bad_freq_words[tox]= zip(*bad_freqs[tox].most_common(mostcommon))[0]
    return CleanFreqWords, bad_freq_words
    
    


def TagText(text, return_length = False):
    tokens = ['You'] + nltk.word_tokenize(text.decode('ascii','ignore'))
    tags = nltk.pos_tag(tokens)
    tags = zip(*tags)[1][1:]
    tags = np.array(tags, dtype = '|S2')
    return ' '.join(tags)

def check_overwrite(outdf, col, overwrite_cols):
    if col in overwrite_cols:
        raise KeyError
    else: 
        return outdf[col]

def ProcessText(infile, outfile = None, nrows = 10000, use_toxic_words = True, Tagger = None, TagFile = None, importantwords = False, reusefile = None, overwrite_cols = [], keras_params = {'trainfile': 'train.csv', 'max_features': 2000, 'maxlen': 100}):
    df = pd.read_csv(infile, nrows = nrows)
    outdf = df.drop('comment_text', axis = 1)
    if (reusefile != None): 
        outdf = pd.read_csv(reusefile)
    df['comment_text'] = df.comment_text.astype('string')
    def Length(): return df["comment_text"].apply(lambda x: len(x))    
    def NumSentences(): return df["comment_text"].apply(lambda x: len(re.findall('[a-zA-Z][\s]*[\.\?\!]', x, re.DOTALL)))
    def AvgSentenceLength(): return outdf['Length'].div(outdf['NumSentences'])
    def NumWords(): return df["comment_text"].apply(lambda x: len(str(x).split()))
    def UniqueWords(): return df["comment_text"].apply(lambda x: len(set(str(x).split()))).div(outdf['NumWords'])
    def TotalCaps(): return  df["comment_text"].apply(lambda x: len(re.findall('[A-Z]', x))).div(outdf['NumSentences'])
    def ConsecutiveCaps(): return df["comment_text"].apply(lambda x: len(''.join(re.findall('[A-Z]+[A-Z]+', x)))).div(outdf['Length'])
    def FirstNoCaps(): return df["comment_text"].apply(lambda x: len(re.findall('[\.\?\!]\s*[a-z]', x))).div(outdf['NumSentences'])
    def ExPoints(): return df["comment_text"].apply(lambda x: len(re.findall('\!', x))).div(outdf['NumSentences'])
    def SpecialChars(): return df["comment_text"].apply(lambda x: len(re.findall('[@#\$%\^&\*]', x)))
    def ProperLinks(): return df["comment_text"].apply(lambda x: len(re.findall('[{\[].*\|.*[}\]]', x)))
    def u(): return  df["comment_text"].apply(lambda x: len(re.findall('\Au\s', x.lower())) + len(re.findall('\su\s', x.lower())) + len(re.findall('\nu\s', x.lower()))  + len(re.findall('\su\n', x.lower())) + len(re.findall('\su\Z', x.lower()))  + len(re.findall('[\.\,\!\?]u\s', x.lower()))).div(outdf['Length'])
    def i(): return df["comment_text"].apply(lambda x: len(re.findall('\Ai\s', x)) + len(re.findall('\si\s', x)) + len(re.findall('\ni\s', x))  + len(re.findall('\si\n', x)) + len(re.findall('\si\Z', x)) + len(re.findall('[\.\,\!\?]i\s', x))).div(outdf['Length'])
    def IPAddress(): return  df["comment_text"].apply(lambda x: find_IP_address(x))
    for col, func in zip(['Length', 'NumSentences', 'AvgSentenceLength', 'NumWords', 'UniqueWords', 'TotalCaps', 'ConsecutiveCaps', 'FirstNoCaps', 'ExPoints', 'SpecialChars', 'ProperLinks', 'u', 'i', 'IPAddress'], [Length, NumSentences, AvgSentenceLength, NumWords, UniqueWords, TotalCaps, ConsecutiveCaps, FirstNoCaps, ExPoints, SpecialChars, ProperLinks, u, i, IPAddress]):
        try:
            outdf[col] = check_overwrite(outdf, col, overwrite_cols)
        except KeyError:
            outdf[col] = func()

    if use_toxic_words:
        for toxic_word in np.append(toxic_words, ['you', 'very']): 
            try:
                outdf[toxic_word] = check_overwrite(outdf, toxic_word, overwrite_cols)
            except KeyError:
                outdf[toxic_word] = df["comment_text"].apply(lambda x: len(re.findall(toxic_word, x.lower())))
        try:
            outdf['mispelled_fuck'] = check_overwrite(outdf, 'mispelled_fuck', overwrite_cols)
        except KeyError:
            outdf['mispelled_fuck'] = df["comment_text"].apply(lambda x: len(re.findall('f.ck', x.lower())) +  len(re.findall('fu.k', x.lower())) + len(re.findall('/{3,6}[fuck\*]', x.lower())) + len(re.findall('phu/{1,2}[ck]', x.lower())) )
        for toxic_word in isolated_words:
            try:
                outdf['{}_isolated'.format(toxic_word)] = check_overwrite(outdf, '{}_isolated'.format(toxic_word), overwrite_cols)
            except KeyError:
                outdf['{}_isolated'.format(toxic_word)] = df["comment_text"].apply(lambda x: len(re.findall('[\s!@#\$%\^&\*\(\),\.:;\'"\[\]-]{}[\s!@#\$%\^&\*\(\),\.:;\'"\[\]-]'.format(toxic_word), x.lower())))
        categories = ['bad_words', 'bigot_words', 'forum_action_words', 'debate_words', 'troll_words', 'toxic_words']
        for category, cat_words in zip(categories, [bad_words, bigot_words, forum_action_words, troll_words, toxic_words]):
            outdf[category] = outdf[cat_words].sum(axis=1)
        if importantwords:
            outdf = outdf.loc[:, np.in1d(outdf.columns, toxic_words[np.in1d(toxic_words, important_words, invert = True)], invert = True)]
    if Tagger != None:
        print 'At Tagging'
        if (not('Tags' in overwrite_cols)) & (TagFile != None):
            tagdf = pd.read_csv(TagFile)
            df['Tags'] = tagdf.Tags.astype('string')
        else:
            df['Tags'] = df["comment_text"].apply(lambda x: TagText(x))
        for POS, POStag in zip(['Nouns', 'Verbs', 'Adjectives', 'Adverbs', 'Pronouns', 'Determiners', 'Prepositions', 'WHs', 'CCs'], ['NN', 'VB', 'JJ', 'RB', 'PR', 'DT', 'IN', 'W[A-Z]', 'CC']):
            try:
                outdf[POS] = check_overwrite(outdf, POS, overwrite_cols)
            except KeyError:
                outdf[POS] = df["Tags"].apply(lambda x: len(re.findall(POStag, x))).div(outdf['NumWords'])
        def Imperatives(): return df["Tags"].apply(lambda x: len(re.findall('\AVB', x)) + len(re.findall('\. VB', x))).div(outdf['NumSentences'])
        def PronounRefs(): return df["Tags"].apply(lambda x: len(re.findall('PR JJ', x)) + len(re.findall('PR NN', x)) + len(re.findall('PR RB', x)) + len(re.findall('PR VB NN', x)) + len(re.findall('PR VB JJ', x)) + len(re.findall('PR VB DT', x))).div(outdf['NumSentences'])
        def CommaClauses(): return df["Tags"].apply(lambda x: len(re.findall(', JJ NN', x)) + len(re.findall(', NN', x)) + len(re.findall(', JJ JJ', x))).div(outdf['NumSentences'])
        def ConsecutiveNouns(): return df["Tags"].apply(lambda x: len(re.findall('NN NN', x))).div(outdf['NumWords'])
        def ConsecutiveAdjs(): return df["Tags"].apply(lambda x: len(re.findall('JJ JJ', x))).div(outdf['NumWords'])
        def BadEndings(): return df["Tags"].apply(lambda x: len(re.findall('IN\Z', x)) + len(re.findall('DT\Z', x)) + len(re.findall('IN \.', x)) + len(re.findall('DT \.', x))).div(outdf['NumSentences'])
        def NumTags(): return df['Tags'].apply(lambda x: len(re.findall('[A-Z][A-Z]', x)))
        for col, func in zip(['Imperatives', 'PronounRefs', 'CommaClauses', 'ConsecutiveNouns','ConsecutiveAdjs','BadEndings','NumTags'],[Imperatives, PronounRefs, CommaClauses, ConsecutiveNouns,ConsecutiveAdjs,BadEndings,NumTags]):
            try:
                outdf[col] = check_overwrite(outdf, col, overwrite_cols)
            except KeyError:
                outdf[col] = func()

        if keras_params != None:
            print 'at keras'
            train = pd.read_csv(keras_params['trainfile'])
            tokenizer = Tokenizer(num_words = max_features)
            list_sentences_train = train["comment_text"].fillna("_na_").values
            tokenizer.fit_on_texts(list(list_sentences_train))
            list_sentences_df = df["comment_text"].fillna("_na_").values
            list_tokenized_df = tokenizer.texts_to_sequences(list_sentences_df)
            X_train = pad_sequences(list_tokenized_df, maxlen = maxlen)
            def UncommonWords(): return outdf['NumTags'].sub(np.count_nonzero(X_train, axis = 1))
            def WordCommonness(): return np.sum(X_train, axis = 1)*1./np.count_nonzero(X_train, axis = 1)
            for col, func in zip(['UncommonWords','WordCommonness'],[UncommonWords,WordCommonness]):
                try:
                    outdf[col] = check_overwrite(outdf, col, overwrite_cols)
                except KeyError:
                    outdf[col] = func()
            outdf.WordCommonness.fillna(max_features, inplace = True)

        if TagFile != None: 
            tagdf = df[['id','Tags']]
            tagdf.to_csv(TagFile, index = False)
    if outfile != None: outdf.to_csv(outfile, index = False)
    return outdf
