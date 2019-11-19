import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import linear_model
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.model_selection import train_test_split
from pandas import DataFrame
import matplotlib.pyplot as pyplot

%pylab

#%%
# Загружаем набор данных Ирисы:
iris = datasets.load_iris()
# Смотрим на названия переменных
print(iris.feature_names)
# Смотрим на данные, выводим 10 первых строк:
print(iris.data[:10])
# Смотрим на целевую переменную:
print(iris.target_names)
# print(iris.target)

#%%
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
import seaborn as sns
sns.pairplot(iris_frame[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)', 'name']], hue='name')
#%%
iris_frame[['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']].corr()

#%%
import seaborn as sns
corr = iris_frame[['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']].corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
  ax = sns.heatmap(corr, mask=mask, square=True, cbar=False, annot=True, linewidths=.5)

#%%
train_data, test_data, train_labels, test_labels = train_test_split(iris_frame[['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']], iris_frame['target'], test_size = 0.3, random_state = 0)
# визуально проверяем, что получившееся разбиение соответствует нашим ожиданиям:
print(train_data)
print(test_data)
print(train_labels)
print(test_labels)

#%%
from scipy import polyval, stats
fit_output = stats.linregress(iris_frame[['petal length (cm)','petal width (cm)']])
slope, intercept, r_value, p_value, slope_std_error = fit_output
print(slope, intercept, r_value, p_value, slope_std_error)


#%%
import matplotlib.pyplot as plt
plt.figure(figsize(5, 5))
plt.plot(iris_frame[['petal length (cm)']], iris_frame[['petal width (cm)']],'o', label='Data')
plt.plot(iris_frame[['petal length (cm)']], intercept + slope*iris_frame[['petal length (cm)']], 'r', linewidth=3, label='Linear regression line')
plt.ylabel('petal width (cm)')
plt.xlabel('petal length (cm)')
plt.legend()
plt.show()

#%%
from sklearn.linear_model import SGDClassifier
train_data, test_data, train_labels, test_labels = train_test_split(iris_frame[['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']], iris_frame[['target']], test_size = 0.3, random_state = 0)
model = SGDClassifier(alpha=0.001, max_iter=100, random_state = 0)
model.fit(train_data, train_labels)
model_predictions = model.predict(test_data)
print(metrics.accuracy_score(test_labels, model_predictions))
print(metrics.classification_report(test_labels, model_predictions))

#%%
