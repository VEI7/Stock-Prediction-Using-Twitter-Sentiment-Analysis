
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import BernoulliNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from matplotlib import pyplot as plt
from matplotlib import font_manager

NB_accuracy_list = []
SVM_accuracy_list = []
DT_accuracy_list = []
KNN_accuracy_list = []
RF_accuracy_list = []
LR_accuracy_list = []

# Stock Prediction
stock_codes = ['AAPL','CSCO','INCT','MSFT']
for stock_code in stock_codes:
    print "Read from text file and prepare data matrix & target matrix...."
    data_Stock = open('../data/train_data/'+stock_code+'.txt', 'r')
    inp_dataStock = []
    stockfiles = np.loadtxt(data_Stock, delimiter=',')

    inp_dataStock = np.array(stockfiles[:,0:-1], dtype='float')
    stock_Y = stockfiles[:,-1]

    X_stock = np.array(inp_dataStock)
    Y_stock = np.array(stock_Y)

    print "Data matrix & target matrix are ready \n"



    def mean(numbers):
        return sum(numbers)/float(len(numbers))


    print "Stock prediction using SVM model...."
    svm_StockPred_accuracy = []
    NB_accuracy = []
    SVM_accuracy = []
    DT_accuracy = []
    KNN_accuracy = []
    RF_accuracy = []
    LR_accuracy = []
    svn_temp = 0

    svm_final_accuracy = 0
    svm_final_precision = 0
    svm_final_recall = 0
    svm_final_fmeasure = 0

    kf1 = KFold(n_splits=5, shuffle=False)
    for train_x, test_x in kf1.split(X_stock):
        X_train, X_test = X_stock[train_x], X_stock[test_x]
        Y_train, Y_test = Y_stock[train_x], Y_stock[test_x]


        # clf = BernoulliNB()
        clf = BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)

        clf.fit(X_train, Y_train)

        predicted_y = clf.predict(X_test)

        # print "Bernoulli NB =" ,accuracy_score(y_test,predicted_y)
        NB_accuracy.append(accuracy_score(Y_test, predicted_y))

        clf_svm = OneVsRestClassifier(LinearSVC(random_state=0))
        clf_svm.fit(X_train, Y_train)

        predicted_svmy = clf_svm.predict(X_test)
        SVM_accuracy.append(accuracy_score(Y_test, predicted_svmy))

        clf_dt = tree.DecisionTreeClassifier()
        clf_dt.fit(X_train, Y_train)
        predicted_dty = clf_dt.predict(X_test)
        DT_accuracy.append(accuracy_score(Y_test, predicted_dty))

        clf_knn = KNeighborsClassifier(n_neighbors=3)
        clf_knn.fit(X_train, Y_train)
        predicted_y = clf_knn.predict(X_test)
        KNN_accuracy.append(accuracy_score(Y_test, predicted_y))

        clf_rf = RandomForestClassifier(max_depth=2, random_state=0)
        clf_rf.fit(X_train, Y_train)
        predicted_y = clf_rf.predict(X_test)
        RF_accuracy.append(accuracy_score(Y_test, predicted_y))

        clf_lr = LogisticRegression(C=1.0, penalty='l1', tol=0.01)
        clf_lr.fit(X_train, Y_train)
        predicted_y = clf_lr.predict(X_test)
        LR_accuracy.append(accuracy_score(Y_test, predicted_y))

    print "Stock prediction for "+stock_code
    print "NB Accuracy =" ,mean(NB_accuracy)
    print "SVM Accuracy =" ,mean(SVM_accuracy)
    print "DT Accuracy =" ,mean(DT_accuracy)
    print "KNN Accuracy =" ,mean(KNN_accuracy)
    print "RF Accuracy =" ,mean(RF_accuracy)
    print "LR Accuracy =" ,mean(LR_accuracy)
    print "\n"
    print "Stock predicted successfully. \n"
    NB_accuracy_list.append(mean(NB_accuracy))
    SVM_accuracy_list.append(mean(SVM_accuracy))
    DT_accuracy_list.append(mean(DT_accuracy))
    KNN_accuracy_list.append(mean(KNN_accuracy))
    RF_accuracy_list.append(mean(RF_accuracy))
    LR_accuracy_list.append(mean(LR_accuracy))

accuracy_dict = {'Naive_Bayes': NB_accuracy_list, 'SVM': SVM_accuracy_list, 'Decision_Tree': DT_accuracy_list, \
                 'KNN': KNN_accuracy_list, 'Ramdom_Forest': RF_accuracy_list, 'Logistic_Regression': LR_accuracy_list}

for k,v in accuracy_dict.items():
    plt.figure(figsize=(10, 5))

    rects = plt.bar(range(len(stock_codes)), v, width=0.5, align="center", fc='lightskyblue', edgecolor='white')
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width()/4., 1.02 * height, '%.5s' % float(height))
    autolabel(rects)

    plt.xticks(range(len(stock_codes)), stock_codes)
    plt.ylim([0, 0.8])
    plt.ylabel("accuracy")
    avg_acc = mean(v)
    plt.title(k + ' - average accuracy: %.5s' % avg_acc)

    plt.savefig("output/" + k + '.png')