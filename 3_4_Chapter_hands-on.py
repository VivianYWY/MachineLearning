
# coding: utf-8

# In[2]:


from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')


# In[3]:


mnist


# In[5]:


X,y = mnist["data"],mnist["target"]


# In[6]:


X.shape


# In[7]:


y.shape


# In[8]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt


# In[9]:


some_digit = X[36000]
some_digit_image = some_digit.reshape(28,28)


# In[11]:


plt.imshow(some_digit_image, cmap=matplotlib.cm.binary,interpolation="nearest")
#plt.axis("off")
plt.show()


# In[12]:


y[36000]


# In[13]:


X_train, X_test, y_train, y_test = X[:60000],X[60000:],y[:60000],y[60000:]


# In[14]:


import numpy as np
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index],y_train[shuffle_index]


# In[15]:


y_train_5 =(y_train==5)
y_test_5 =(y_test==5)


# In[17]:


from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train,y_train_5)


# In[19]:


y_train_5[1]


# In[20]:


sgd_clf.predict([some_digit])


# In[22]:


from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, X_train, y_train_5, cv=3,scoring="accuracy")


# In[23]:


from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)


# In[24]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_train_5, y_train_pred)


# In[27]:


from sklearn.metrics import precision_score, recall_score
precision_score(y_train_5, y_train_pred)


# In[28]:


recall_score(y_train_5, y_train_pred)


# In[29]:


y_scores = sgd_clf.decision_function([some_digit])
y_scores


# In[30]:


threshold = 0
y_some_digit_pred = (y_scores > threshold)


# In[31]:


y_some_digit_pred


# In[32]:


y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")


# In[33]:


from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)


# In[35]:


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0,1])
plot_precision_recall_vs_threshold(precisions,recalls,thresholds)
plt.show()


# In[36]:


from sklearn.metrics import roc_curve
fpr,tpr,thresholds = roc_curve(y_train_5, y_scores)


# In[37]:


from sklearn.metrics import roc_auc_score
roc_auc_score(y_train_5, y_scores)


# In[38]:


from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf,X_train, y_train_5, cv=3,method="predict_proba")


# In[39]:


from sklearn.multiclass import OneVsOneClassifier
ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42))
ovo_clf.fit(X_train, y_train)
ovo_clf.predict([some_digit])


# In[40]:


len(ovo_clf.estimators_)


# In[41]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")


# In[42]:


y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)


# In[43]:


conf_mx = confusion_matrix(y_train, y_train_pred)


# In[44]:


plt.matshow(conf_mx,cmap=plt.cm.gray)
plt.show()


# In[45]:


row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx/row_sums


# In[46]:


np.fill_diagonal(norm_conf_mx,0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.show()


# In[48]:


from sklearn.neighbors import KNeighborsClassifier
y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 ==1)
y_multilabel = np.c_[y_train_large, y_train_odd]
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)


# In[54]:


import numpy.random as rnd
noise = rnd.randint(0,100,(len(X_train),784))
noise1 = rnd.randint(0,100,(len(X_test),784))
X_train_mod = X_train + noise
X_test_mod = X_test + noise1
y_train_mod = X_train
y_test_mod = X_test


# In[52]:


X_train.shape


# In[53]:


len(X_train)


# In[55]:


knn_clf.fit(X_train_mod, y_train_mod)


# In[59]:



clean_digit = knn_clf.predict([X_test_mod[some_index]])
plot_digit(clean_digit)

