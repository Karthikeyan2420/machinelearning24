#svm -support vector Machine
#hyperplae line base classifier



import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

iris=datasets.load_iris()
X=iris.data[:,:2]
y=iris.target

xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.2,random_state=42)
svmcla=SVC(kernel='linear')
svmcla.fit(xtrain,ytrain)
ypred=svmcla.predict(xtest)

acc=accuracy_score(ytest,ypred)
print("Accuracy : ",acc)

xmin,xmax=X[:,0].min()-1,X[:,0].max()+1
ymin,ymax=X[:,1].min()-1,X[:,1].max()+1
xx,yy=np.meshgrid(np.arange(xmin,xmax,0.1),np.arange(ymin,ymax,0.1))
z=svmcla.predict(np.c_[xx.ravel(),yy.ravel()])
z=z.reshape(xx.shape)

plt.contourf(xx,yy,z,alpha=0.4)
plt.scatter(X[:,0],X[:,1],c=y,s=20,edgecolors='k')
plt.show()