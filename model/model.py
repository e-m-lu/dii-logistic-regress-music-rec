import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import train_test_split

def label_map(label):
    for i in range(label.shape[0]):
        print(label[i])
        if label[i]>=50 and label[i]<=60:
            label[i] = 0
        elif label[i]>60 and label[i]<=70:
            label[i] = 1
        elif label[i]>70 and label[i]<=80:
            label[i] = 2
        elif label[i]>80 and label[i]<=90:
            label[i] = 3
        elif label[i]>90 and label[i]<=100:
            label[i] = 4
    return label

def DataProcessing(file_path):
    data = pd.read_csv('Data.csv', encoding='gbk')
    tempo = (data['Tempo']-data['Tempo'].min())/(data['Tempo'].max()-data['Tempo'].min())
    
    genre = data['Genre']
    le = LabelEncoder()
    encoder_result = le.fit_transform(genre)
    label = data['HR perc']

    new_label = label_map(label)

    del_list = ['ID', 'Genre', 'Tempo']
    data2 = data.drop(del_list,axis=1,inplace=False)
    data2.insert(2, 'Genre', encoder_result)
    data2.insert(1, 'Tempo', tempo)
    train_data = data2.drop(['HR perc', 'HR'], axis=1,inplace=False)
    print('Data Processing is done....')
    return train_data, new_label

def model(train_data, new_label):
    train_x, test_x, train_y, test_y = train_test_split(train_data,new_label,test_size=0.5, random_state=0) 
    clf = LogisticRegression()
    clf.fit(train_x, train_y)
    score = clf.score(test_x, test_y)
    return score

def main():
    train_data, new_label = DataProcessing('Data.csv')
    score = model(train_data, new_label)
    print('Accuracy is: ', score)

if __name__ == '__main__':
    main()
