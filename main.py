from flask import Flask, render_template, request
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import numpy as np
app=Flask(__name__)
@app.route('/')
def home():
    return render_template("home.html")
@app.route('/value', methods=['POST','GET'])
def value():
    if(request.method=='POST'):
        cont = request.form['cont']
        newsgroups_train = fetch_20newsgroups(subset='train')
        newsgroups_test = fetch_20newsgroups(subset='test')

        count_vect= CountVectorizer()
        tfidfvectorizer = TfidfTransformer()
        trainvec = count_vect.fit_transform(newsgroups_train.data)  
        traintfidf = tfidfvectorizer.fit_transform(trainvec)

        clf = MultinomialNB()
        clf.fit(traintfidf,newsgroups_train.target)
        vectors_test = count_vect.transform(newsgroups_test.data)
        vectors_test_tfidf=tfidfvectorizer.transform(vectors_test)
        Y_test_pred = clf.predict(vectors_test_tfidf)
        inp=[]
        inp.append(cont)
        print(inp)

        print('Accuracy achieved is ' + str(np.mean(Y_test_pred == newsgroups_test.target)))
        print(metrics.classification_report(newsgroups_test.target, Y_test_pred, target_names=newsgroups_test.target_names)),
        metrics.confusion_matrix(newsgroups_test.target, Y_test_pred)
        

        x_new_count=count_vect.transform(inp)
        x_new_tfidf=tfidfvectorizer.transform(x_new_count)
        Y_test_pred1 = clf.predict(x_new_tfidf)
        output=newsgroups_train.target_names[Y_test_pred1[0]]
        print(output)
        cat="CATEGORY : "
        
    return render_template("home.html",category=cat,output=output)

if __name__=="__main__":
    app.run()