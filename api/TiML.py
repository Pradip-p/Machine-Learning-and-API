import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pickle
import os

def preprocessing(df):
    def get_title(name):
        if '.'in name:
            return name.split(',')[1].split('.')[0].strip()
        else:
            return "unknown"
    def title_map(title):
        if title in ['Mr']:
            return 1
        elif title in ['Master']:
            return 3
        elif title in ['Ms', 'Mlle', 'Miss']:
            return 4
        elif title in ['Mme','Mrs']:
            return 5
        else:
            return 2
    df['title']=df['Name'].apply(get_title).apply(title_map)
    df=df.drop(["PassengerId","Name","Ticket"],axis="columns")
    df["Sex"]=df["Sex"].replace(["male","female"],[0,1])
    df['Age'][df['Age'].isna()]=df['Age'].mean()
    print(df["Age"])
    df["Cabin"]=df["Cabin"].isna() #Nan values changes to boolean
    mf=df['Fare'].mean()
    df['Fare']=df['Fare']>mf
    df['Fare']=df['Fare'].astype(int)
    df=pd.get_dummies(df)
    return df


def training(df):
    df=preprocessing(df)
    y=df["Survived"]
    df.drop("Survived", axis="columns", inplace=True)
    x=df
    dummyRow=pd.DataFrame(np.zeros(len(x.columns)).reshape(1,len(x.columns)), columns=x.columns)
    model=XGBClassifier(max_depth=2,min_child_weight=3, gamma=0,subsample=0.86, reg_alpha=0, n_estimators=125)
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=10)
    model.fit(x,y)
    #print(model.score(x_test,y_test))
    pkl_filename="pickle_model.pkl"
    with open(pkl_filename,'wb') as file:
        pickle.dump(model,file)

# yp=model.predict(x_test)
# print("Survived", sum(yp!=0)) 
# print("not Survived ", sum(yp==0))
# #accuracy_score(y_test,yp)
# cm=confusion_matrix(y_test, yp)
# #import seaborn as sn
# #sn.heatmap(cm,annot=True)
# cm

def pred(ob):
    d1=ob.to_dict()
    df=pd.DataFrame(d1,index=[0])
    df.drop("Survived", axis="columns", inplace=True)
    df=preprocessing(df)
    dummyRow_filename="dummyRow.csv"
    dummyRow_filename=os.path.dirname(__file__)+'/'+dummyRow_filename
    df2=pd.read_csv(dummyRow_filename)
    for c1 in df.columns:
        df2[c1]=df[c1]
    # training(df)
    pkl_filename='./pickle_model.pkl'
    #pkl_filename=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pkl_filename=os.path.join.abspath(os.path.dirname(__file__),pkl_filename)
    # pkl_filename=os.path.dirname(__file__)+'/'+pkl_filename
    print(pkl_filename)
    print("Hello")
    with open(pkl_filename,'rb') as file:
        print(pkl_filename)
        model=pickle.load(file)
        print(model)
    pred=model.predict(df2)
    return pred

if __name__=="__main__":
    df=pd.read_csv("titanic.csv")
    training(df)



