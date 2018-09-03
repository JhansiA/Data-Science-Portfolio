# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 14:46:15 2018

@author: Jhansi
"""

from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import logging

import pandas as pd
import functions
import re
import string
import numpy as np
from rasa_nlu.model import Interpreter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
import MySQLdb

logger = logging.getLogger(__name__)
logging.basicConfig(filename='prediction_pipeline.log',filemode='w', level=logging.DEBUG)
logging.info('Started')

try:    
    # import May dataset# get from db
    language,label_id_in,note_id_in = functions.get_basic_info()
    logging.info('Prediction pipeline started for label: '+str(label_id_in))
    parser,stop_words,configfile_path = functions.lang_change(language)
    
    stopwords = list(stop_words)
    #Add stopwords from database
    stopwords_db = functions.get_stopwords() #get the stop words in a list 
    stopwords.extend(stopwords_db)
    
    punctuations = string.punctuation
    
    Must_have_words = functions.must_have_words()
    words_re = re.compile("|".join(Must_have_words))
    
    pred_data = functions.get_rasa_testing_data()
    #pred_data = pd.read_csv("population_split_may2018.csv")
    pred_data.drop_duplicates(inplace=True)
    if (len(pred_data)!=0):
        pred_dataset =  functions.split_sentence(pred_data,words_re)
        mention_cleaned = []
        for i in pred_dataset.Mention:
            mention_cleaned.append(functions.spacy_clean_sent(i,parser,stopwords,punctuations))
        pred_dataset['Mention_cleaned'] = mention_cleaned
        pred_dataset['Mention_cleaned'].replace('', np.nan, inplace=True)
        pred_dataset.dropna(subset=['Mention_cleaned'], inplace=True)
        logging.info('Unlabeled dataset shape is {} for label '.format(label_id_in))
        logging.info('Loading rasa model for label: '+str(label_id_in))
        first_model_path,second_model_path,model_intent,rasa_output_table,model_threshold,final_output = functions.get_model_summary()
        interpreter = Interpreter.load('./models/default/'+first_model_path)  
        logging.info('Loaded rasa model for label: '+str(label_id_in))
        #Will take long time (>1hr depends on data)
        logging.info('Prediction started using rasa model for label: '+str(label_id_in))
        pred_dataset_count = pred_dataset.Mention_cleaned.count()
        df_rasa_pred_output = pd.DataFrame(columns=['id','mention','confidence','intent'])
        for i in range(pred_dataset_count):
            extract_chatlog = pred_dataset.Mention_cleaned.iloc[i]
            if extract_chatlog != '':
                classify_chat = interpreter.parse(extract_chatlog)
                id_pred = pred_dataset.Id.iloc[i]
                df1 = functions.predict_text_intent(classify_chat, id_pred)    
                df_rasa_pred_output = df_rasa_pred_output.append(pd.DataFrame(data = df1), ignore_index=True)
        logging.info('Prediction done using rasa model for label {} and output shape is {}'.format(label_id_in,df_rasa_pred_output.shape))
        #We can save model1 prediction results if we want
        
        model2_data = df_rasa_pred_output[df_rasa_pred_output['intent']==model_intent]
        logging.info('Second model started for label {} and input shape is {}'.format(label_id_in,model2_data.shape))
        if (len(model2_data)!=0):
            test = model2_data['mention']
            model_threshold = 0.84
            vocab_dict = functions.get_vocabulary()
            logging.info('Vocabulary is loaded for label: '+str(label_id_in))
            countvectorizer = CountVectorizer(preprocessor=functions.cleanText, tokenizer= functions.spacy_tokenizer, ngram_range=(1,4), vocabulary=vocab_dict)
            test_vec = countvectorizer.fit_transform(test)
            logging.info('Loading second model for label: '+str(label_id_in))
            grid_search_clf = joblib.load(second_model_path+'.pkl')
            logging.info('Prediction started using second model for label: '+str(label_id_in))
            y_pred = (grid_search_clf.predict_proba(test_vec)[:,1] >= model_threshold).astype(int) 
            logging.info('Prediction ended using second model for label: '+str(label_id_in))
            model2_data['target'] = y_pred
            df_final_output = model2_data.merge(pred_dataset,how = 'inner',on =None, left_on = ['id','mention'],right_on=['Id','Mention_cleaned'])
            df_final_output.Original_mention = df_final_output.Original_mention.str.encode('utf-8')
            
            if(len(df_final_output)!=0):
                conn = MySQLdb.connect(host = "hostnumber",user = "root",passwd = "",db = "database")
                cursor = conn.cursor()
                cursor.execute("call create_final_result("+str(label_id_in)+","+str(note_id_in)+",'"+str(final_output)+"')")
                cursor.executemany(" INSERT INTO "+final_output+" (comment_id, mention,target,confidence) VALUES (%s,%s,%s,%s) ", zip(df_final_output['id'],df_final_output['Original_mention'],df_final_output['target'],df_final_output['confidence']) )
                conn.close()
            else:
                logging.info('Final output is empty for selected label: '+str(label_id_in))
            #save this model2 into database
            logging.info('Prediction pipeline is done for label: '+str(label_id_in))
        else:
            logging.error("Second model can't be run because rasa model output is empty for label:"+str(label_id_in))
except Exception as ex:
    logging.error('Error occured in prediction pipeline for label: '+str(label_id_in)+' and the error: '+str(ex))
    