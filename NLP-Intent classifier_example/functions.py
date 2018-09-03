# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 17:42:56 2018

@author: Jhansi

It is having all helping functions for model 1 and model 2
"""
from rasa_nlu.training_data import load_data
from rasa_nlu import config
from rasa_nlu.model import Trainer
#from rasa_nlu.model import Interpreter, Metadata
import pandas as pd
import string
import yaml
import MySQLdb
import os
#logger = logging.getLogger(__name__)
#logging.basicConfig(filename='rasamodel.log',filemode='w', level=logging.DEBUG)
conn = MySQLdb.connect(host = "hostnumber",
                           user = "root",
                           passwd = "",
                           db = "database")
language = 'en'
label_id_in = 5 #get from webpage
note_id_in = 5 #get from webpage
start_date = '2018-05-01'#get from webpage
end_date = '2018-05-15'#get from webpage
must_have_filter = 1 #check with Sam and Bryanna

#get language from database and update lower case into variable, by default English
def get_basic_info():
    return language,label_id_in,note_id_in

def get_stopwords():
    list_to_show = 'stopword'
    stopwords_data =  pd.read_sql_query("call get_data('"+list_to_show+"',"+str(label_id_in)+")", conn)
    return stopwords_data['stopword'].tolist()

def lang_change(language):
    if language == 'en':
        from spacy.lang.en import English
        from spacy.lang.en.stop_words import STOP_WORDS
        parser = English()
        file="\config_files\config_spacy_en.yaml"
        configfile_path=os.getcwd()+file
    elif language == 'de':
        from spacy.lang.de import German
        from spacy.lang.de.stop_words import STOP_WORDS
        parser = German()  
        file="\config_files\config_spacy_de.yaml"
        configfile_path=os.getcwd()+file
    elif language == 'es':
        from spacy.lang.es import Spanish
        from spacy.lang.es.stop_words import STOP_WORDS
        parser = Spanish()
        file="\config_files\config_spacy_es.yaml"
        configfile_path=os.getcwd()+file
    elif language == 'pt':
        from spacy.lang.pt import Portuguese
        from spacy.lang.pt.stop_words import STOP_WORDS
        parser = Portuguese()
        file="\config_files\config_spacy_pt.yaml"
        configfile_path=os.getcwd()+file
    elif language == 'fr':
        from spacy.lang.fr import French
        from spacy.lang.fr.stop_words import STOP_WORDS
        parser = French()
        file="\config_files\config_spacy_fr.yaml"
        configfile_path=os.getcwd()+file
    elif language == 'it':
        from spacy.lang.it import Italian
        from spacy.lang.it.stop_words import STOP_WORDS
        parser = Italian()
        file="\config_files\config_spacy_it.yaml"
        configfile_path=os.getcwd()+file
    elif language == 'nl':
        from spacy.lang.nl import Dutch
        from spacy.lang.nl.stop_words import STOP_WORDS
        parser = Dutch()    
        file="\config_files\config_spacy_nl.yaml"
        configfile_path=os.getcwd()+file
         
    return parser,STOP_WORDS,configfile_path    

#language = get_language()
parser,stop_words,configfile_path  = lang_change(language)

stopwords = list(stop_words)
punctuations = string.punctuation

def must_have_words():
    list_to_show = 'must_have'
    Must_have_data = pd.read_sql_query("call get_data('"+list_to_show+"',"+str(label_id_in)+")", conn)
    return Must_have_data['must_have_word'].tolist() #['repaired','service'] # will get form database in a list

def get_rasa_training_data():
    training_data = pd.read_sql_query("call get_train_data_tensorflow("+str(label_id_in)+",'"+language+"')", conn)
    training_data=  training_data[['Mention','Intent']]
    return training_data

def get_rasa_testing_data():
    test_data =  pd.read_sql_query("call get_unlabeled("+str(label_id_in)+",'" +start_date+"','"+end_date+"')", conn)   
    return test_data 

def get_validated_data():
    validate_data =pd.read_sql_query("call get_validated_data("+str(label_id_in)+","+str(must_have_filter)+")", conn) 
    return validate_data

def get_vocabulary():
    vocabulary_data = pd.read_sql_query("select vocabulary, feature_mtx_index from tc_vocabulary where label_id ="+str(label_id_in), conn)
    vocab_dict= dict([(key,value) for key,value in zip(vocabulary_data.vocabulary, vocabulary_data.feature_mtx_index)])
    return vocab_dict

def split_sentence(df,words_re):
    split_data = pd.DataFrame(columns= ['Id','Mention','Original_mention'])
    for index,row in df.iterrows():
        try:
            word_count = len(row["Mention"].split(' '))
            if word_count > 30:
                list = row["Mention"].split('.')
                list_org = row["original_mention"].split('.')
                for i,j in zip(list,list_org):
                    if words_re.search(i):
                        split_data = split_data.append({'Id': row['Comment_id'],'Mention':i,'Original_mention':j}, ignore_index=True)
            else:
                split_data = split_data.append({'Id': row['Comment_id'],'Mention':row['Mention'],'Original_mention':row['original_mention']}, ignore_index=True)
        except:
            continue
    return split_data  

def spacy_clean_sent(sentence,parser,stopwords,punctuations):
    word_list = parser(sentence)
    word_list = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in word_list]
    word_list = [word for word in word_list if word not in stopwords and word not in punctuations]
    new_sent = " ".join(word_list)
    #mention_cleaned.append(new_sent)
    return new_sent #mention_cleaned

def convert_excel_rasa_nlu(training_data):
    results = []
    common_examples = []
    x = []
    for index, row in training_data.iterrows():
        x = (row['Intent'],row['Mention_cleaned'],[])
        results.append(x)
    for result in results:
        keys = ['intent', 'text', 'entities']
        common_examples.append(dict(zip(keys, result)))
    rasa_nlu_data1 = {"regex_features" : [], "entity_synonyms": [],"common_examples": []}
    rasa_nlu_data1["common_examples"] = common_examples
    dataformat = dict(rasa_nlu_data = rasa_nlu_data1)
    return(dataformat)
    
#Create a data cleansing function
def clean_text(text):     
    return text.strip(string.punctuation).lower()

def get_model_summary():
    summary = pd.read_sql_query("call get_model_note("+str(label_id_in)+")", conn)
    first_model_path = summary['first_model_path'].tolist()[0]
    second_model_path = summary['second_model_path'].tolist()[0]
    model_intent = summary['intent'].tolist()[0]
    model_threshold = summary['model_threshold'].tolist()[0]
    rasa_output_table = summary['output_table'].tolist()[0]
    final_output = summary['final_output'].tolist()[0]
    return first_model_path,second_model_path,model_intent,rasa_output_table,model_threshold,final_output

def train_nlu(data, configs, model_dir,first_model_path):
    #print('in')
    training_data = load_data(data)
    trainer = Trainer(config.load(configs))
    trainer.train(training_data)
    model_directory = trainer.persist(model_dir, fixed_model_name = first_model_path)
    return model_directory

#Use model to predict data in excel format into predefined format
#fix the model so that the processing power can be run faster
#df_rasa_output = pd.DataFrame(columns=['id','mention','confidence','intent'])
def predict_text_intent(intent_result, id_pred):
    #global df_rasa_output
    if intent_result['intent']['name'] in ['spam']:
        #check if it is a spam comment, if yes then append into the final dataframe
        df1 = pd.DataFrame(columns=['id','mention','confidence','intent'])
        extract_text = intent_result['text']
        extract_confidence = intent_result['intent']['confidence']
        extract_intent = intent_result['intent']['name']
        if extract_confidence > 0.5:
            row = [id_pred, extract_text, extract_confidence, extract_intent]
            df1.loc[len(df1)] = row
            df1.append(row, ignore_index=True)
        #df_rasa_output = df_rasa_output.append(pd.DataFrame(data = df1), ignore_index=True)
    else:
        df1 = pd.DataFrame(columns=['id','mention','confidence','intent'])
        count_intent_range = len(intent_result['intent_ranking'])
        intent_ranking_result = intent_result['intent_ranking']
        for i in range(count_intent_range):
            string = []
            index_range = i
            string = intent_ranking_result[index_range]
            extract_text = intent_result['text']
            extract_confidence = string['confidence']
            extract_intent = string['name']
            if extract_confidence > 0.5:
                row = [id_pred, extract_text, extract_confidence, extract_intent]
                df1.loc[len(df1)] = row
                df1.append(row, ignore_index=True)
            #df_rasa_output = df_rasa_output.append(pd.DataFrame(data = df1), ignore_index=True)
    return df1#df_rasa_output.drop_duplicates(inplace=True)  

# Function for tokenizing giant string(stemming, removing stopwords and punctuation)
def spacy_tokenizer(sentence):
    word_list = parser(sentence)
    word_list = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in word_list]
    word_list = [word for word in word_list if word not in stopwords and word not in punctuations]
    return word_list   

# basic function to clean the text
def cleanText(text):
    text = text.strip().replace("\n", " ").replace("\r", " ")
    text = text.lower()
    return text