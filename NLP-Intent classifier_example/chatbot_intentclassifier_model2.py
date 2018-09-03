# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 16:42:33 2018

@author: Jhansi

Using user validated data for model2 to create classifier model
"""
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import logging
from builtins import zip

import pandas as pd 
import numpy as np
from  sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.externals import joblib
import MySQLdb
import functions 

logging.basicConfig(filename='sklearnmodel.log',filemode='w', level=logging.DEBUG)
logging.info('Started')

try:
    
    user_accuracy = 0 #get from webpage
    
    language,label_id_in,note_id_in = functions.get_basic_info()
    logging.info('Second model is started for label: '+str(label_id_in))
    parser,stop_words,configfile_path = functions.lang_change(language)
    
    stopwords = list(stop_words)
    #Add stopwords from database
    stopwords_db = functions.get_stopwords() #get the stop words in a list 
    stopwords.extend(stopwords_db)
    
    #get from database : column names are Intent,Mention,Target
    validated_data = functions.get_validated_data()
    if (len(validated_data)!=0):
        logging.info('Validation data is available for label: '+str(label_id_in))
        validated_data = validated_data[['Mention','Target']]
        #validated_data = pd.read_excel("Copy of validated_data_jan_April_repair_tat_sam.xlsx")
        # drop duplicated columns
        validated_data.drop_duplicates(inplace=True)
        #print(validated_data.Target.value_counts())
        if len(validated_data.Target.value_counts())==1:
            logging.error("There should be atleast 2 classes in the validation data for label: "+str(label_id_in))
        X = validated_data.Mention
        y = validated_data.Target
        X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.20, random_state=42,stratify = y)
        
        # create a vocabulary
        count_vec = CountVectorizer(stop_words=stopwords, analyzer='word', 
                                    ngram_range=(1, 4), max_df=1.0, min_df=1)
        count_vec_fit = count_vec.fit(X)
        vocab_list = count_vec_fit.vocabulary_
        # To Save the vocabulary list into database # see for better way later
        list_vocab = list(vocab_list.keys())
        list_index = list(vocab_list.values())
        df_vocab = pd.DataFrame(columns=['label_id','vocabulary','feature_index'])
        df_vocab['vocabulary'],df_vocab['feature_index'] = list_vocab,list_index
        df_vocab['label_id'] = label_id_in
        
        countvectorizer = CountVectorizer(preprocessor=functions.cleanText, tokenizer= functions.spacy_tokenizer, ngram_range=(1,4), vocabulary=vocab_list)
        
        X_train_vec = countvectorizer.fit_transform(X_train)
        X_test_vec = countvectorizer.fit_transform(X_test)
        
        # Creating dataframe to store the evaluation metrics for each machine learning algorithm
        Eval_metrics = pd.DataFrame(columns= ['Model','Class','Precision','Recall','F1-score'])
        
        #Help function for evaluation metrics
        def classification_report_df(report, name):
            lines = report.split('\n')
            row_data = lines[-4].split('      ')
            dict_1 = {'Model':name,'Class':'1','Precision' : float(row_data[2]),'Recall' : float(row_data[3]),'F1-score' : float(row_data[4]) }
        #     row_data = lines[-2].split('      ')
        #     dict_total = {'Model':name,'Class':'Total','Precision' : float(row_data[1]),'Recall' : float(row_data[2]),'F1-score' : float(row_data[3]) }
            return dict_1#,dict_total
        
        def train_and_test(steps, X_train, X_test, y_train, y_test,pipeline_name):
            """
            Trains and tests the pipeline with the given steps. 
            :param steps:       List of operations inside the pipeline.
            :param X_train:     Training data
            :param X_test:      Training labels
            :param y_train:     Testing data
            :param y_test:      Testing labels
            :return:            Trained model
            """
            global Eval_metrics
            pipeline = Pipeline(steps)
            folds = 5
            xval_score = cross_val_score(pipeline, X_train, y_train, cv=folds)
            
            xv_min = np.min(xval_score)
            xv_max = np.max(xval_score)
            xv_mean = np.mean(xval_score)
            
            logging.info('{} fold Cross Validation Score for {} for label : {} is <{:.2f}, {:.2f}>; µ={:.2f}'.format(folds,pipeline_name ,label_id_in,xv_min, xv_max, xv_mean))
            pipeline = pipeline.fit(X_train, y_train)
            logging.info('Score on test set for label {} : {:.2f}'.format(label_id_in,pipeline.score(X_test, y_test)))
            # validation test
            preds = pipeline.predict(X_test)
            #Evaluating the model
            report = metrics.classification_report(y_test, preds)
            Evaluations_1 = classification_report_df(report,pipeline_name)
            Eval_metrics = Eval_metrics.append(Evaluations_1,ignore_index= True) 
            
            return pipeline
        
        def best_classifier():
            clf = SVC(kernel='linear', probability=True, class_weight='balanced', random_state=42)
            steps = ([('feature_selection', SelectFromModel(GradientBoostingClassifier(learning_rate=0.15, max_depth=4))),
                    ("classifier", clf)])
            train_and_test(steps, X_train_vec, X_test_vec, y_train, y_test,'pipeline_SVC')
        
            clf = RandomForestClassifier(max_depth=25, max_features=20, min_samples_split=5,n_estimators =300, random_state=42, n_jobs=-1)
            steps = ([('feature_selection', SelectFromModel(GradientBoostingClassifier(learning_rate=0.15, max_depth=4))),
                    ("classifier", clf)])
            train_and_test(steps, X_train_vec, X_test_vec, y_train, y_test,'pipeline_RFC')
        
            clf = DecisionTreeClassifier(criterion='entropy', max_depth=15, max_features=5,max_leaf_nodes=5, random_state=42)
            steps = ([('feature_selection', SelectFromModel(GradientBoostingClassifier(learning_rate=0.15, max_depth=4))),
                    ("classifier", clf)])
            train_and_test(steps, X_train_vec, X_test_vec, y_train, y_test,'pipeline_DTC')
        
            clf = MultinomialNB(alpha=2)
            steps = ([('feature_selection', SelectFromModel(GradientBoostingClassifier(learning_rate=0.15, max_depth=4))),
                    ("classifier", clf)])
            train_and_test(steps, X_train_vec, X_test_vec, y_train, y_test,'pipeline_NB')
        
            best_clf = Eval_metrics['Model'].iloc[Eval_metrics['Recall'].values.argmax()]
            print(best_clf)
            return best_clf
        
        best_clf = best_classifier()
        logging.info('Best classifier for second model is:{} for label: {}'.format(best_clf,label_id_in))
        def best_pipeline(best_clf):
            if best_clf == "pipeline_SVC":
                clf = SVC()
                param_grid = {'clf__kernel': ['linear'],
                              'clf__probability':[True],
                              'clf__class_weight':['balanced'],
                              'clf__random_state': [42]
                              }
            elif best_clf == "pipeline_RFC":
                clf = RandomForestClassifier()
                param_grid = {'clf__min_samples_split': [3, 5, 10], 
                              'clf__n_estimators' : [100, 300],
                              'clf__max_depth': [3, 5, 15, 25],
                              'clf__max_features': [3, 5, 10, 20]}
            elif best_clf == "pipeline_DTC":
                clf = DecisionTreeClassifier()
                param_grid = {'clf__max_leaf_nodes' : [3,5,7,10,],
                              'clf__max_depth': [3, 5, 15, 25],
                              'clf__max_features': [3, 5, 10, 20],
                              'clf__criterion':['entropy']}
            elif best_clf == "pipeline_DTC":
                clf = MultinomialNB()
                param_grid = {'clf__alpha':[1.0,1.5,0.5,2]}    
            
            steps = [('feature_selection', SelectFromModel(GradientBoostingClassifier(learning_rate=0.15, max_depth=4))), ('clf', clf)]
            pipe_line = Pipeline(steps)
            return pipe_line,param_grid
        
        pipe_line,param_grid = best_pipeline(best_clf)
        
        def grid_search_wrapper():
            """
            fits a GridSearchCV classifier using refit_score for optimization
            prints classifier performance metrics
            """
            logging.info('Gridsearch for best classifier is started for label: '+str(label_id_in))
            skf = StratifiedKFold(n_splits=10)
            grid_search = GridSearchCV(pipe_line, param_grid,cv=skf)
            grid_search.fit(X_train_vec, y_train)
        
            # make the predictions
            y_pred = grid_search.predict(X_test_vec)
        
            print(grid_search.best_params_)
        
            # confusion matrix on the test data.
            print(pd.DataFrame(confusion_matrix(y_test, y_pred),
                         columns=['pred_neg', 'pred_pos'], index=['neg', 'pos']))
            print(classification_report(y_test, y_pred))
            logging.info('Gridsearch for best classifier is finished for label: '+str(label_id_in))
            return grid_search
        
        grid_search_clf = grid_search_wrapper()
        
        first_model_path,second_model_path,model_intent,rasa_output_table,model_threshold,final_output = functions.get_model_summary()
        joblib.dump(grid_search_clf, second_model_path+'.pkl')
        logging.info('Second model is saved for label: '+str(label_id_in))
        def optimal_threshold(model_pipeline, X_test, y_test, user_accuracy = 0):
            pred_prob = model_pipeline.predict_proba(X_test)
            precision, recall, threshold = precision_recall_curve(y_test.values, pred_prob[:,1], pos_label= 1)
            df =  pd.DataFrame(columns= ['Precision','Recall','Threshold','Minimum_value'])
            df['Precision'],df['Recall'],df['Threshold'],df['Minimum_value'] = precision[:-1],recall[:-1],threshold, abs(precision[:-1]-recall[:-1])
            if user_accuracy == 0:
                opt_threshold = round(df['Threshold'].iloc[df['Minimum_value'].values.argmin()],2)
                
            elif df['Precision'].max() >= user_accuracy:
                df = df[df['Precision']>= user_accuracy]
                opt_threshold = round(df['Threshold'].iloc[df['Minimum_value'].values.argmin()],2)
            else:
                user_accuracy = df['Precision'].max()
                opt_threshold = round(df['Threshold'].iloc[df['Minimum_value'].values.argmin()],2)
            return opt_threshold
        
        threshold = optimal_threshold(grid_search_clf,X_test_vec,y_test, user_accuracy)
        logging.info('Optimal threshold is: {:2f} for label : {} '.format(threshold,label_id_in))
        
        conn = MySQLdb.connect(host = "hostnumber",user = "root",passwd = "",db = "database")
        cursor = conn.cursor()
        #save opt_threshold to database
        cursor.execute("call save_threshold("+str(label_id_in)+","+str(threshold)+")")
        #Clean the existing vocabulary for this model 
        if(conn):
            cursor.execute("call clean_vocab("+str(label_id_in)+")")
        else:
            conn = MySQLdb.connect(host = "hostnumber",user = "root",passwd = "",db = "database")
            cursor = conn.cursor()    
            cursor.execute("call clean_vocab("+str(label_id_in)+")")
        #Insert new vocabulary
        if(conn):
            cursor.executemany(" INSERT INTO tc_vocabulary (label_id, vocabulary,feature_mtx_index) VALUES (%s,%s,%s) ", zip(df_vocab['label_id'],df_vocab['vocabulary'],df_vocab['feature_index']) )
        else:
            conn = MySQLdb.connect(host = "hostnumber",user = "root",passwd = "",db = "database")
            cursor = conn.cursor()
            cursor.executemany(" INSERT INTO tc_vocabulary (label_id, vocabulary,feature_mtx_index) VALUES (%s,%s,%s) ", zip(df_vocab['label_id'],df_vocab['vocabulary'],df_vocab['feature_index']) )
            
        conn.close()
        logging.info('Second model is completed for label: '+str(label_id_in))
    else:
        logging.info('Validated data is not available for second model for label: '+str(label_id_in))
except Exception as ex:
    logging.error('Error occured in second model for label: '+str(label_id_in)+' and the error: '+str(ex))

