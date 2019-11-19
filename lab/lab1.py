#%%
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import linear_model
from sklearn.cluster import KMeans
from sklearn import metrics

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split
from pandas import DataFrame
import matplotlib.pyplot as pyplot
import seaborn as sns

%pylab inline

#%%
# Загружаем набор данных Ирисы:
iris = datasets.load_iris()
# Добавляем их в DataFrame
iris_frame = DataFrame(iris.data)
# Делаем имена колонок такие же, как имена переменных:
iris_frame.columns = iris.feature_names
# Добавляем столбец с целевой переменной:
iris_frame['target'] = iris.target
# Для наглядности добавляем столбец с сортами:
iris_frame['name'] = iris_frame.target.apply(lambda x : iris.target_names[x])
# Смотрим, что получилось:
iris_frame

#%%
# Строим гистограммы по каждому признаку:
pyplot.figure(figsize(15, 20))
plot_number = 0
for feature_name in iris['feature_names']:
  for target_name in iris['target_names']:
    plot_number += 1
    pyplot.subplot(4, 3, plot_number)
    pyplot.hist(iris_frame[iris_frame.name == target_name][feature_name])
    pyplot.title(target_name)
    pyplot.xlabel('cm')
    pyplot.ylabel(feature_name[:-4])

#%%
sns.pairplot(iris_frame[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)', 'name']], hue='name')
#%%
# Корреляции по всем переменным в таблице
iris_frame[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']].corr()

#%%
# Корреляции по классам в таблице
iris_frame.groupby(['name'])[['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']].corr()

#%%
corr = iris_frame[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']].corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
  ax = sns.heatmap(corr, mask=mask, square=True, cbar=False, annot=True, linewidths=.5)

#%%
raw_data = iris.data[:,:2]
test_size = 0.3

def use(algorithm, raw_data, targets, test_size, x_label = 'x', y_label = 'y'):

  train_data, test_data, train_labels, test_labels = \
    train_test_split(raw_data, targets, test_size = test_size, random_state = 2)

  algorithm.fit(train_data, train_labels)

  l_bound = test_data[:, 0].min() - 0.5
  r_bound = test_data[:, 0].max() + 0.5
  b_bound = test_data[:, 1].min() - 0.5
  t_bound = test_data[:, 1].max() + 0.5

  h_spacing = np.arange(l_bound, r_bound, 0.05)
  v_spacing = np.arange(b_bound, t_bound, 0.05)

  x_coord, y_coord = np.meshgrid(h_spacing, v_spacing)

  prediction = algorithm.predict(
    np.c_[x_coord.ravel(), y_coord.ravel()]
  ).reshape(x_coord.shape)

  accuracy = algorithm.predict(test_data)
  print(str(metrics.accuracy_score(test_labels, accuracy)))

  pyplot.figure()

  pyplot.title(algorithm.__class__.__name__)
  pyplot.xlabel(x_label)
  pyplot.ylabel(y_label)

  pyplot.pcolormesh(x_coord, y_coord, prediction, cmap = pyplot.cm.Paired)
  pyplot.scatter(
    test_data[:, 0],
    test_data[:, 1],
    c=test_labels,
    edgecolors='black',
    cmap=pyplot.cm.Paired
  )

# Linear Discriminant
linear_discriminant = LinearDiscriminantAnalysis()
use(linear_discriminant, raw_data, iris.target, test_size)

quadratic_discriminant = QuadraticDiscriminantAnalysis()
use(quadratic_discriminant, raw_data, iris.target, test_size)

logistic_regression = LogisticRegression(C = 1e5, solver = 'lbfgs', multi_class = 'multinomial')
use(logistic_regression, raw_data, iris.target, test_size)

svm_linear = SVC(kernel='linear')
use(svm_linear, raw_data, iris.target, test_size)

svm_quadratic = SVC()
use(svm_quadratic, raw_data, iris.target, test_size)


# %%
