# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 11:47:55 2018

@author: Jhansi

Using rasa model to get intent based on message
"""

from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import logging
from builtins import zip

from rasa_nlu.model import Interpreter#, Metadata
import pandas as pd
import json
import string
import re
import MySQLdb
import functions
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(filename='rasamodel.log',filemode='w', level=logging.DEBUG)
logging.info('Started')

try:
    #get language from webpage and update lower case into variable, by default English
    language,label_id_in,note_id_in = functions.get_basic_info() 
    logging.info('Rasa model started for label: '+str(label_id_in))
    parser,stop_words,configfile_path = functions.lang_change(language)
    
    stopwords = list(stop_words)
    #Add stopwords from database
    stopwords_db = functions.get_stopwords() #get the stop words in a list 
    stopwords.extend(stopwords_db)
    
    punctuations = string.punctuation
    
    Must_have_words = functions.must_have_words()
    words_re = re.compile("|".join(Must_have_words))
    
    # importing training data # get from database # make sure column names are Intent, Mention, if ID exists then Comment_id
    training_data = functions.get_rasa_training_data() 
    if(len(training_data)!=0):
        
        #training_data = pd.read_excel("training_samples2.xlsx",encoding ="utf-8")  
        # change the type of intent column as category
        training_data.Intent = training_data.Intent.astype('category')
        
        #Removed this below logic as it should be handled by database 
        #training_data.Mention = training_data.Mention.str.lower()
        #training_data = training_data[training_data.Mention.str.contains('service')] 
        mention_cleaned = []
        for i in training_data.Mention:
            mention_cleaned.append(functions.spacy_clean_sent(i,parser,stopwords,punctuations))
        
        training_data['Mention_cleaned'] = mention_cleaned
        training_data['Mention_cleaned'].replace('', np.nan, inplace=True)
        training_data.dropna(subset=['Mention_cleaned'], inplace=True)
        
        logging.info('Training data shape: {} for label: {} '.format(training_data.shape,label_id_in))
        #Create a data cleansing function
        def clean_text(text):     
            return text.strip(string.punctuation).lower()
        
        #Convert excel training data into rasa_nlu format
        with open('rasa_dataset_'+str(label_id_in)+'.json', 'w') as fp:
            json.dump(functions.convert_excel_rasa_nlu(training_data),fp)
        
        first_model_path,second_model_path,model_intent,rasa_output_table,model_threshold,final_output = functions.get_model_summary()
            
        # def run_nlu():
        #     interpreter = Interpreter.load('./models/default/rasaTLnlu')    
        # will take few minutes
        #if __name__ == '__main__':
        logging.info('Rasa training started for label: '+str(label_id_in))
        model_directory=functions.train_nlu('rasa_dataset_'+str(label_id_in)+'.json', configfile_path, './models/',first_model_path)
        logging.info(model_directory+' rasa model created for label: '+str(label_id_in))
        interpreter = Interpreter.load(model_directory)
        logging.info('Rasa training ended for label: '+str(label_id_in))
        # import unlabeled dataset # get from database # columns Comment_id, Mention
        logging.info('Test data started for label: '+str(label_id_in))
        test_data = functions.get_rasa_testing_data() 
        test_data.drop_duplicates(inplace=True)
        if(len(test_data)!=0): 
            #test_data = pd.read_excel("unlabeled_population_split_service_Jan_May.xlsx") 
            test_dataset =  functions.split_sentence(test_data,words_re)
            
            mention_cleaned = []
            for i in test_dataset.Mention:
                mention_cleaned.append(functions.spacy_clean_sent(i,parser,stopwords,punctuations))
            test_dataset['Mention_cleaned'] = mention_cleaned
            test_dataset['Mention_cleaned'].replace('', np.nan, inplace=True)
            test_dataset.dropna(subset=['Mention_cleaned'], inplace=True)
            logging.info('Test data shape: {} for label: {} '.format(test_dataset.shape,label_id_in))
            #Will take long time (>1hr depends on data)
            logging.info('Unlabeled data prediction started using rasa for label: '+str(label_id_in))
            test_dataset_count = test_dataset.Mention_cleaned.count()
            df_rasa_output = pd.DataFrame(columns=['id','mention','confidence','intent'])
            for i in range(test_dataset_count):
                extract_chatlog = test_dataset.Mention_cleaned.iloc[i]
                if extract_chatlog != '':
                    classify_chat = interpreter.parse(extract_chatlog)
                    id_pred = test_dataset.Id.iloc[i]
                    df1 = functions.predict_text_intent(classify_chat, id_pred)    
                    df_rasa_output = df_rasa_output.append(pd.DataFrame(data = df1), ignore_index=True)
            logging.info('Unlabeled data prediction ended using rasa for label: '+str(label_id_in))
            
            df_rasa_output = df_rasa_output[df_rasa_output['intent']==model_intent]
            df_rasa = df_rasa_output.merge(test_dataset,how = 'inner',on =None, left_on = ['id','mention'],right_on=['Id','Mention_cleaned'])
            #df_rasa_output = df_rasa_output[df_rasa_output['intent']=='bad_repair']
            logging.info('Rasa output shape {} for label: {} '.format(df_rasa_output.shape,label_id_in))
            df_rasa.Original_mention = df_rasa.Original_mention.str.encode('utf-8')
            if(len(df_rasa)!=0):
                conn = MySQLdb.connect(host = "hostnumber",user = "root",passwd = "",db = "database")
                cursor = conn.cursor()
                cursor.executemany(" INSERT INTO "+rasa_output_table+" (comment_id, mention,confidence) VALUES (%s,%s,%s) ", zip(df_rasa['id'],df_rasa['Original_mention'],df_rasa['confidence']) )
                conn.close()
            else:
                logging.info('Rasa output is empty for selected label: '+str(label_id_in))
            logging.info('Rasa model is done for label: '+str(label_id_in))
        else:
            logging.info('No unlabeled data available for rasa model label: '+str(label_id_in))
    else:
        logging.info('No training data available for rasa model label: '+str(label_id_in))
except Exception as ex:
    logging.error('Error occured in rasa model for label: '+str(label_id_in)+' and the error: '+str(ex))

            
             