# Классификация чисел

## Задания

1. Проделать все шаги из туториала; 
2. Подрефакторить скрипт написанный на python3 для обучения нейсросети распознаванию рукописных чисел. (перенести все import наверх и т.д.);
3. Поэксперементировать с созданием MLP классификатора:
    - Добавить дополнительные скрытые слои;
    - Поизменять размер скрытых слоёв;
    - Поизменять точность;
    - Поизменять кол-во итераций;
    - Попробовать различные функции активации;
    - Попробовать сделать выводы.

## Туториал

### Импорт библиотек и набора данных

```python
# importing the hand written digit dataset
from sklearn import datasets

# digit contain the dataset
digits = datasets.load_digits()

# dir function use to display the attributes of the dataset
print(dir(digits))
```

### Функция для печати набора изображений

```python
# outputting the picture value as a series of numbers
print(digits.images[0])
```
Исходные цифры имели гораздо более высокое разрешение, и оно было уменьшено при подготовке набора данных для scikit-learn, чтобы система машинного обучения могла распознавать эти цифры проще и быстрее.

### Вывод чисел на графике

```python
# importing the matplotlib libraries pyplot function
import matplotlib.pyplot as plt
# defining the function plot_multi

def plot_multi(i=0):
    nplots = 16
    fig = plt.figure(figsize=(15, 15))
    for j in range(nplots):
        plt.subplot(4, 4, j+1)
        plt.imshow(digits.images[i+j], cmap='binary')
        plt.title(digits.target[i+j])
        plt.axis('off')
    # printing the each digits in the dataset.
    plt.show()

plot_multi()
```

### Создание набора данных

```python
# converting the 2 dimensional array to one dimensional array
y = digits.target
x = digits.images.reshape((len(digits.images), -1))

# Very first 1000 photographs and
# labels will be used in training.
x_train = x[:1000]
y_train = y[:1000]

# The leftover dataset will be utilised to
# test the network's performance later on.
x_test = x[1000:]
y_test = y[1000:]
```

### Использование многослойного классификатора персептронов и обучение модели

```python
# importing the MLP classifier from sklearn
from sklearn.neural_network import MLPClassifier

# calling the MLP classifier with specific parameters
mlp = MLPClassifier(hidden_layer_sizes=(15,),
                    activation='logistic',
                    alpha=1e-4, 
                    solver='sgd',
                    tol=1e-4, 
                    random_state=1,
                    learning_rate_init=.1,
                    verbose=True,
                    )

# trainig model
mlp.fit(x_train, y_train)
```

### Отображение результатов обучения на графике и оценка модели

```python
# importing the accuracy_score from the sklearn
from sklearn.metrics import accuracy_score

# drawing plot
fig, axes = plt.subplots(1, 1)
axes.plot(mlp.loss_curve_, 'o-')
axes.set_xlabel("number of iteration")
axes.set_ylabel("loss")
plt.show()

# calculate predictions
predictions = mlp.predict(x_test)

# calculating the accuracy with y_test and predictions
print(accuracy_score(y_test, predictions))
```
