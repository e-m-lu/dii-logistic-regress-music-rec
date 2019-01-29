import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import train_test_split

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np


# image can only show 2D, but feature space is 9D, so we only used the first 2D

def label_map(label):
    for i in range(label.shape[0]):
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
    data = pd.read_csv('data.csv', encoding='gbk')
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

def plot_decision_regions(X, y, test_idx=None, resolution=0.02):
    # draw decision boundary: X = feature, y = label, classifier，test_idx = training set sequence number
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
 
    # plot decision surface
    print('X.........')
    # print(X)
    print(X.shape)
    X = np.array(X)
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1   # the first feature uses feature range as x-axis
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1   # the second feature uses feature range as y-axis
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))  # resolution = meshing granularity, xx1 and xx2 have the same dimension
    clf = LogisticRegression()
    clf.fit(np.array(X[:,:2]), y)
    Z = clf.predict(np.array([xx1.ravel(), xx2.ravel()]).T)   
    # classifier, ravel = array flattening, Z uses each pair of the two features to predict
    Z = Z.reshape(xx1.shape)   #  Z = column vector
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)  
    # contourf(x,y,z): x and y are 2 same-length 1D array，z = 2D array -> z value for each xy pair
    # fill the area between the contour lines with different colors
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
 
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)   # use full dataset: plot data points of different categories as coordinates (x,y), in different colors
    plt.show()
    # highlight test samples
    if test_idx:
        # plot all samples
        if not versiontuple(np.__version__) >= versiontuple('1.9.0'):
            X_test, y_test = X[list(test_idx), :], y[list(test_idx)]
            warnings.warn('need NumPy 1.9.0 or newer')
        else:
            X_test, y_test = X[test_idx, :], y[test_idx]   # X_test takes training set features, y_test takes training set labels
 
        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    alpha=1.0,
                    linewidths=1,
                    marker='o',
                    s=55, label='test set')   # c sets colors, test set uses different types of dots to represent categories, in same color


def main():
    train_data, new_label = DataProcessing('data.csv')
    plot_decision_regions(train_data, new_label)

if __name__ == '__main__':
    main()
