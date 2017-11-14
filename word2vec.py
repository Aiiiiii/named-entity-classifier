# coding: utf-8


from collections import Counter
from itertools import product
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
import urllib.request
from nltk.corpus import brown
from gensim.models import Word2Vec


def download_data():
 
    url = 'https://www.dropbox.com/s/bqitsnhk911ndqs/train.txt?dl=1'
    urllib.request.urlretrieve(url, 'train.txt')
    url = 'https://www.dropbox.com/s/s4gdb9fjex2afxs/test.txt?dl=1'    
    urllib.request.urlretrieve(url, 'test.txt')
    
    
def read_data(filename):
    
    '''import csv
    from itertools import groupby
    #train_file = '/Users/aijiang/Downloads/train.txt'

    with open(filename, 'r') as f:
      lists = []
      for line in f:
        if '-DOCSTART-' in line:
          continue
        word = [line.strip()]
        lists.extend(word)
    line_list = []
    line_list = [list(group) for k, group in groupby(lists, lambda x: x == '') if not k]

    line = []
    for lin in line_list:
      entry = []
      for etr in lin:
        token, pos, pctag, netag = etr.split()
        entry.append((token, pos, pctag, netag))
      line.append(entry)
    return line '''
    
    all_sentences = []
    this_sentence = []
    for line in open(filename):
        if '-DOCSTART-' in line:
            continue
        if len(line.strip())==0:
            #next sentence 
            if len(this_sentence) > 0:
                all_sentences.append(this_sentence)
                this_sentence = []
        else:
            this_sentence.append(tuple(line.strip().split()))
    if len(this_sentence) > 0:
        all_sentences.append(this_sentence)
    #print ('read data: ' + str(all_sentences[:2]))
    return all_sentences   


def make_feature_dicts(data,
                       w2v_model,
                      token=True,
                       caps=True,
                       pos=True,
                       chunk=True,
                       context=True,
                       w2v=True):
    """
    Create feature dictionaries, one per token. Each entry in the dict consists of a key (a string)
    and a value of 1.
   
    """

    #a numpy array of NER tags (strings), one per token.
    '''token = []
    lables = []
    for lin in data:
      for etr in lin:
          lables.append(etr[-1])
          token.append(etr[0])
    lables = np.array(lables)
    token_number = len(token)
    import copy
    count=0
    finalFeature = []
    for s in data:
      sentenceFeature=[]
      flag=count
      for etr in s:
        entryFeature = {}
        token = etr[0]
        if token:
          entryFeature['tok=' + token.lower()] = 1
        if caps:
          if token[0].isupper():
            entryFeature['is_caps'] = 1
          else:
            entryFeature.update({})
        if pos:
          entryFeature['pos=' + etr[1]] = 1
        if chunk:
          entryFeature['chunk=' + etr[2]]= 1
        temp = copy.deepcopy(entryFeature)
        sentenceFeature.append(entryFeature)
        finalFeature.append(temp)
        count += 1
      if context:
        if sentenceFeature and len(sentenceFeature) != 1:
          #print(sentenceFeature[1])
          for x in sentenceFeature[1]:
            finalFeature[flag]['next_%s'%x] = 1
          for i in range(1,len(sentenceFeature)-1):
            for x in sentenceFeature[i+1]:
              finalFeature[flag+i]['next_%s'%x] = 1
            for x in sentenceFeature[i-1]:
              finalFeature[flag+i]['prev_%s'%x] = 1
          for x in sentenceFeature[len(sentenceFeature)-2]:
            finalFeature[flag+(len(sentenceFeature)-1)]['prev_%s'%x] = 1

    return finalFeature, lables'''
    all_dicts = []
    all_labels = []
    #model = Word2Vec(brown.sents(), min_count=5, size=50, window=5)
    for sentence in data:
        sentence_dicts = []
        for i, word in enumerate(sentence):
            feats = {}
            
            if token:
                feats['tok=%s'%word[0].lower()] = 1
            if caps and word[0][0].isupper():
                feats['is_caps'] = 1
            if pos:
                feats['pos=%s'%word[1]] = 1
            if chunk:
                feats['chunk=%s'%word[2]] = 1
            if w2v:
                if word[0] in w2v_model.wv.vocab:
                    for i, v in enumerate(w2v_model.wv[word[0]]):
                        feats['w2v_%d'%(i+1)] = v
            sentence_dicts.append(feats)
            all_labels.append(word[-1])
        if context:
            new_dicts = []
            #add prev/next features
            for i, d in enumerate(sentence_dicts):
                new_dict = dict(d)
                if i > 0:
                    for k, v in sentence_dicts[i-1].items():
                        new_dict['prev_%s'%k] = v
                if i < len(sentence_dicts) - 1:
                    for k, v in sentence_dicts[i+1].items():
                        new_dict['next_%s'%k] = v
                new_dicts.append(new_dict)
            sentence_dicts = new_dicts
        all_dicts.extend(sentence_dicts)
    return all_dicts, np.array(all_labels)
   


def confusion(true_labels, pred_labels):
    """
    Create a confusion matrix, where cell (i,j)
    is the number of tokens with true label i and predicted label j.
    """

    labels = np.unique(true_labels)
    pred = np.unique(pred_labels)
    combine = list(product(labels,pred))
    true_pred = list(zip(true_labels, pred_labels))
    count_dicts = {}
    count_dicts.update({tup: true_pred.count(tup) for tup in combine})

    df = pd.DataFrame()
    for row_label in labels:
      rowdata= {}
      for col_label in pred:
        for key, value in count_dicts.items():
          if key == (row_label,col_label):
            rowdata[col_label] = value
      df = df.append(pd.DataFrame.from_dict({row_label:rowdata}, orient='index'))

    return df

def evaluate(confusion_matrix):
    """
    Compute precision, recall, f1 for each NER label.
    
    """
    labels = list(confusion_matrix.columns)
    scores = np.zeros((3, len(labels)))
    for i, label in enumerate(labels):
        npred = confusion_matrix[label].sum()
        ntrue = confusion_matrix.ix[label].sum()
        pr = 0 if npred.any() == 0 else confusion_matrix[label][label] / npred
        re = 0 if ntrue.any() == 0 else confusion_matrix[label][label] / ntrue
        f1 = 0 if (pr + re == 0) else (2 * pr * re) / (pr + re)
        scores[0,i] = pr
        scores[1,i] = re
        scores[2,i] = f1
    return pd.DataFrame(scores, index=['precision', 'recall', 'f1'], columns=labels)
    '''df_cm = confusion_matrix
    df_evl = pd.DataFrame()
    evaluate = ['precision', 'recall', 'f1']
    sum_row = df_cm.sum(axis=1)
    sum_column = df_cm.sum(axis=0)
    labels = sorted(list(df_cm.index.values))

    for row_label in evaluate:
      rowdata= {}
      for col_label in labels:
        if row_label == 'precision':
          rowdata[col_label] = df_cm.at[col_label, col_label] / sum_column[col_label] #precision value = tp/tp+fp
        if row_label == 'recall':
          rowdata[col_label] = df_cm.at[col_label, col_label] / sum_row[col_label]  #recall value = tp/tp+fn
        if row_label == 'f1':
          p = df_cm.at[col_label, col_label] / sum_column[col_label]
          r = df_cm.at[col_label, col_label] / sum_row[col_label]
          rowdata[col_label] = 2 * (p*r)/(p+r)        #f1 value = 2 * (p*r)/(p+r)
      df_evl = df_evl.append(pd.DataFrame.from_dict({row_label:rowdata}, orient='index'))
    return df_evl'''
def average_f1s(evaluation_matrix):
    
    '''
    df = evaluation_matrix
    labels = list(df.columns)
    #print (labels)
    select = df.loc[['f1'],[label for label in labels if label != 'O']]
    #print (select)
    mean = select.mean(axis=1)
    return mean
    '''
    vals = evaluation_matrix.ix['f1'][:4]
    return 0 if vals.sum() == 0 else vals.mean()

def evaluate_combinations(train_data, test_data):
    """
  	Run 16 different settings of the classifier, 
  	corresponding to the 16 different assignments to the
  	parameters to make_feature_dicts:
  	caps, pos, chunk, context"""
    '''
    A = [False,True]
    settings = list(product(A,A,A,A))
    df = pd.DataFrame()
    for n, setting in enumerate(settings):
      #print(setting)
      s_caps = setting[0]
      s_pos = setting[1]
      s_chunk = setting[2]
      s_context = setting[3]
      dicts, labels = make_feature_dicts(train_data,
                                     True,
                                     s_caps,
                                     s_pos,
                                     s_chunk,
                                     s_context)

      
      vec = DictVectorizer()
      X = vec.fit_transform(dicts)
      clf = LogisticRegression()
      clf.fit(X, labels)
      npram = np.shape(clf.coef_)[0]*np.shape(clf.coef_)[1]
      test_dicts, test_labels = make_feature_dicts(test_data,
                                                 True,
                                                 s_caps,
                                                 s_pos,
                                                 s_chunk,
                                                 s_context) 
      X_test = vec.transform(test_dicts)
      preds = clf.predict(X_test)
      confusion_matrix = confusion(test_labels, preds)
      evaluation_matrix = evaluate(confusion_matrix)
      avgf1 = average_f1s(evaluation_matrix)

      rowdata = {'f1': float(avgf1), 'n_params': npram, 'caps': s_caps, 'pos': s_pos, 'chunk': s_chunk, 'context': s_context }
      df = df.append(pd.DataFrame.from_dict({n: rowdata}, orient='index'))
      df = df.sort_values(by=['f1'], ascending=[False])
    #print (df)
    return df
    '''
    results = []
    for flags in product([False, True], repeat = 5):
        dicts, labels = make_feature_dicts(train_data,
                                           w2v_model,
                                           token=True,
                                           caps=flags[0],
                                            pos=flags[1],
                                            chunk=flags[2],
                                           context=flags[3],
                                           w2v=flags[4])
        vec = DictVectorizer()
        X = vec.fit_transform(dicts)
        clf = LogisticRegression()
        clf.fit(X,labels)
        nprams = len(clf.coef_.flatten())
        test_data = read_data('test.txt')
        test_dicts, test_labels = make_feature_dicts(test_data,
                                           w2v_model,
                                           token=True,
                                           caps=flags[0],
                                            pos=flags[1],
                                            chunk=flags[2],
                                           context=flags[3],
                                           w2v=flags[4])
        
        X_test = vec.transform(test_dicts)
        preds = clf.predict(X_test)
        confusion_matrix = confusion(test_labels, preds)
        evaluation_matrix = evaluate(confusion_matrix)
        f1 = average_f1s(evaluation_matrix)
        results.append([f1, nprams] + list(flags))
    return pd.DataFrame(results,
                        columns=['f1', 'n_params', 'caps', 'pos', 'chunk', 'context', 'w2v']).sort_values('f1', ascending=False)
        
        


if __name__ == '__main__':
  
    download_data()
    train_data = read_data('train.txt')
    w2v_model = Word2Vec(brown.sents(), min_count=5, size=50, window=5)
    dicts, labels = make_feature_dicts(train_data,
                                       w2v_model,
                                      token=True,
                                       caps=True,
                                       pos=True,
                                       chunk=True,
                                       context=True,
                                       w2v=True)
    #print(dicts[:3])

    vec = DictVectorizer()
    X = vec.fit_transform(dicts)
    print('training data shape: %s\n' % str(X.shape))

    fo = open("output1.txt", "w")
    fo.write('training data shape: %s\n\n' % str(X.shape))

    clf = LogisticRegression()
    clf.fit(X, labels)


    test_data = read_data('test.txt')
    test_dicts, test_labels = make_feature_dicts(test_data,
                                                 w2v_model,
                                                token=True,
                                                 caps=True,
                                                 pos=True,
                                                 chunk=True,
                                                 context=True,
                                                 w2v=True)            
    X_test = vec.transform(test_dicts)
    print('testing data shape: %s\n' % str(X_test.shape))
    fo.write('testing data shape: %s\n\n' % str(X_test.shape))

    preds = clf.predict(X_test)
    #print(test_labels)
    #print (preds)
    confusion_matrix = confusion(test_labels, preds)
    print('confusion matrix:\n%s\n' % str(confusion_matrix))
    fo.write('confusion_matrix: \n%s\n\n' % str(confusion_matrix))

    evaluation_matrix = evaluate(confusion_matrix)
    print('evaluation matrix:\n%s\n' % str(evaluation_matrix))
    fo.write('evaluation_matrix: \n%s\n\n' % str(evaluation_matrix))

    print('average f1s: %f\n' % average_f1s(evaluation_matrix))
    fo.write('average f1s: %f\n\n' % average_f1s(evaluation_matrix))

    combo_results = evaluate_combinations(train_data, test_data)
    print('combination results:\n%s' % str(combo_results))
    fo.write('combination results:\n%s' % str(combo_results))
    fo.close()

