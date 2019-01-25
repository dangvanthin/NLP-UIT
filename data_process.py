from wordsegment import load, segment
from spellchecker import SpellChecker
import pandas as pd, re,string,os,io
import codecs, spacy, html, itertools
import pickle
from ftfy import fix_text
import nltk, csv, sys
import numpy as np
# Preprocessing data
def Escaping_HTML_characters(sent):
    sent_new = html.unescape(sent)
    return sent_new
    
nlp = spacy.load('en_core_web_sm')
contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "can not", 
                   "can't've": "cannot have", "'cause": "because", "could've": "could have", 
                   "couldn't": "could not", "couldn't've": "could not have","didn't": "did not", 
                   "doesn't": "does not", "don't": "do not", "hadn't": "had not", 
                   "hadn't've": "had not have", "hasn't": "has not", "haven't": "have not", 
                   "he'd": "he would", "he'd've": "he would have", "he'll": "he will", 
                   "he'll've": "he will have", "he's": "he is", "how'd": "how did", 
                   "how'd'y": "how do you", "how'll": "how will", "how's": "how is", 
                   "I'd": "I would", "I'd've": "I would have", "I'll": "I will", 
                   "I'll've": "I will have","I'm": "I am", "I've": "I have", 
                   "i'd": "i would", "i'd've": "i would have", "i'll": "i will", 
                   "i'll've": "i will have","i'm": "i am", "i've": "i have", 
                   "isn't": "is not", "it'd": "it would", "it'd've": "it would have", 
                   "it'll": "it will", "it'll've": "it will have","it's": "it is", 
                   "let's": "let us", "ma'am": "madam", "mayn't": "may not", 
                   "might've": "might have","mightn't": "might not","mightn't've": "might not have", 
                   "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", 
                   "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", 
                   "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not",
                   "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", 
                   "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", 
                   "she's": "she is", "should've": "should have", "shouldn't": "should not", 
                   "shouldn't've": "should not have", "so've": "so have","so's": "so as", 
                   "this's": "this is",
                   "that'd": "that would", "that'd've": "that would have","that's": "that is", 
                   "there'd": "there would", "there'd've": "there would have","there's": "there is", 
                       "here's": "here is",
                   "they'd": "they would", "they'd've": "they would have", "they'll": "they will", 
                   "they'll've": "they will have", "they're": "they are", "they've": "they have", 
                   "to've": "to have", "wasn't": "was not", "we'd": "we would", 
                   "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", 
                   "we're": "we are", "we've": "we have", "weren't": "were not", 
                   "what'll": "what will", "what'll've": "what will have", "what're": "what are", 
                   "what's": "what is", "what've": "what have", "when's": "when is", 
                   "when've": "when have", "where'd": "where did", "where's": "where is", 
                   "where've": "where have", "who'll": "who will", "who'll've": "who will have", 
                   "who's": "who is", "who've": "who have", "why's": "why is", "why're":"why are","why'd":"why did", "who've":"who have","which's":"which is",
                       "where've":"where have","where's":"where is","I'm'o":"I am going to","I'm'a":"I am about to",  
                   "why've": "why have", "will've": "will have", "won't": "will not", 
                   "won't've": "will not have", "would've": "would have", "wouldn't": "would not", 
                   "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
                   "y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                   "you'd": "you would", "you'd've": "you would have", "you'll": "you will", 
                   "you'll've": "you will have", "you're": "you are", "you've": "you have", "pls":"please", "'tis": "it is", "'twas":"it was",
                   "o'clock	": "of the clock", "ne'er":"never","gon't":"go not","gonna":"going to","gimme":"give me","everyone's":"everyone is",
                   "e'er":"ever","daren't":"dare not","'cause	":"because","gotta":"got to"} 


## Add feature
def rule_filter(sent):
      keywords = ["it would be nice","would be nice","please add","it would be great to","please allow","please make","it would be great if","would be great to","it would be great","be great to have","i would like to have","i suggest","i would like to see","be nice to have","add an option to","have an option","there should be","have an option","it would be very","i would like to be","i suggest","i would like to see","please provide","it would be good","add support","it would be helpful","be really useful","user should be able","should be add"]
      label = 0
      keyword_match = any(elem in keywords for elem in sent)
      if keyword_match == True:
          label = 1
      return label

def spacy_cleaner(text):
    apostrophe_handled = text
    expanded = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in apostrophe_handled.split(" ")])
    parsed = nlp(expanded)
    final_tokens = []
    for t in parsed:
        if t.is_punct or t.is_space or t.like_num or t.like_url or str(t).startswith('@'):
            pass
        else:
            if t.lemma_ == '-PRON-':
                final_tokens.append(str(t))
            else:
                sc_removed = re.sub("[^a-zA-Z]", '', str(t.lemma_))
                if len(sc_removed) > 1:
                    final_tokens.append(sc_removed)
    joined = ' '.join(final_tokens)
    spell_corrected = re.sub(r'(.)\1+', r'\1\1', joined)
    return spell_corrected

def Split_Attached_Words(sent):
    sent = " ".join(re.findall('[A-Z][^A-Z]*', sent))
    return sent
def normalText(sent):
    sent = re.sub('\\s+',' ', sent)
    patURL = r"(http|www|https)\S+"
    sent = re.sub(patURL,'website',sent)
    sent = re.sub('\.+','.',sent)
    sent = re.sub('\\s+',' ',sent)
    return sent

def editSent(sent):
    sent = ''.join(''.join(s)[:2] for _, s in itertools.groupby(sent))
    return sent

load()
spell = SpellChecker()
def editSegment(sent):
    s = ""
    for word in sent.split(' '):
        if len(word) > 10:
              temp = segment(word)
              for item in temp:
                  s += spell.correction(item) + " "
        else:
            s += spell.correction(word) + " "
    return s.strip()
def clean_doc(doc):
    doc = fix_text(doc)
    doc = normalText(doc)
    doc = Escaping_HTML_characters(doc)
    doc = spacy_cleaner(doc)
  #  doc = Split_Attached_Words(doc)
    
    # Lowercase
    doc = doc.lower()
    # Removing multiple whitespaces
    doc = re.sub(r"\?", " \? ", doc)
    # Remove numbers
    doc = re.sub(r"\b[0-9]+\b", " num ", doc)
    doc = doc.replace("num num", " num ")
    doc = re.sub("\\s+"," ", doc)
    # Split in tokens
    # Remove punctuation
    for punc in string.punctuation:
        doc = doc.replace(punc,' ')
    doc = re.sub('\\s+',' ',doc)
    s = editSent(doc).strip()
    s = editSegment(s).strip()
    return s

    
# read dataset and write output to file 
class Data():
    def __init__(self,stt,text,label):
        self.stt = stt
        self.label = label
        self.text = text


def read_Test_csv(data_path):
    file_reader = csv.reader(open(data_path,"rt", errors="ignore",encoding="utf-8"), delimiter=',')
    sent_list = []

    for row in file_reader:
        id = row[0]
        sent = row[1]
        sent_list.append((id,sent))
    return sent_list

def read_Train_csv(data_path):
    file_reader = csv.reader(open(data_path,"rt", errors="ignore",encoding="utf-8"), delimiter=',')
    sent_list = []

    for row in file_reader:
        id = row[0]
        sent = row[1]
        label = row[2]
        sent_list.append((id,sent,label))
    return sent_list


def write_csv(sent_list, label_list, out_path):
        filewriter = csv.writer(open(out_path, "w+",errors="ignore",encoding="utf-8"))
        count = 0
        for ((id, sent), label) in zip(sent_list, label_list):
                filewriter.writerow([id, sent, label])

def to_category_vector(label):
    vector = np.zeros(2).astype(np.float32)
    i = int(label)
    vector[i] = 1.0
    return vector

if __name__ == '__main__':
    data_path = '/home/thindv/ShareTask/dataset/Training.csv'
    trainData = read_Train_csv(data_path)
    document_X_train = list()
    document_Y_train = list()
    document_X_train_raw = list()
    for data in trainData:
        document_X_train.append(clean_doc(data[1]))
        document_X_train_raw.append(data[1])
        document_Y_train.append(to_category_vector(data[2]))
    
    
    print("Number of X samples: ", len(document_X_train))
    print("Number of Y samples: ", len(document_Y_train))
    PIK = "document_X_train.dat"
    with open(PIK, "wb") as f:
        pickle.dump(document_X_train, f)
        
    PIK2 = "document_Y_train.dat"
    with open(PIK2, "wb") as f:
        pickle.dump(document_Y_train, f)
    
    PIK3 = "document_X_train_raw.dat"
    with open(PIK3, "wb") as f:
        pickle.dump(document_X_train_raw, f)
    print("Data processing done ....")