from loadLibraries import *

# global variables
# Test options and evaluation metric
seed = 7
scoring = 'accuracy'


def load_dataset():
    # Load dataset
    # url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    # ds = pandas.read_csv(url, names=names)
    ds = pandas.read_csv('data/iris.data.csv', names=names)
    return(ds)


def check_dataset():
    # shape
    print(dataset.shape)
    # head
    print(dataset.head(20))
    # descriptions
    print(dataset.describe())
    # class distribution
    print(dataset.groupby('class').size())


def visualize_dataset():
    # Univariate plots to better understand each attribute.
    dataset.plot(kind='box', subplots=True, layout=(2, 2),
                 sharex=False, sharey=False)
    plt.show()
    # histograms
    dataset.hist()
    plt.show()

    # Multivariate plots to better understand the relationships between attributes.
    # scatter plot matrix
    scatter_matrix(dataset)
    plt.show()


def create_validation_dataset():
    # Split-out validation dataset
    array = dataset.values
    X = array[:, 0:4]
    Y = array[:, 4]
    validation_size = 0.20
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(
        X, Y, test_size=validation_size, random_state=seed)
    return(X_train, X_validation, Y_train, Y_validation)


def build_models(compare_algorithms=True):
    # Spot Check Algorithms
    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC()))
    # evaluate each model in turn
    results = []
    names = []
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        cv_results = model_selection.cross_val_score(
            model, X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)

    if compare_algorithms:
        # Compare Algorithms
        fig = plt.figure()
        fig.suptitle('Algorithm Comparison')
        ax = fig.add_subplot(111)
        plt.boxplot(results)
        ax.set_xticklabels(names)
        plt.show()


def make_predictions_knn():
    # Make predictions on validation dataset
    print('KNN predicts : ')
    knn = KNeighborsClassifier()
    knn.fit(X_train, Y_train)
    predictions = knn.predict(X_validation)
    print(accuracy_score(Y_validation, predictions))
    print(confusion_matrix(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))
    return


def make_predictions_SVM():
    # Make predictions on validation dataset
    print('SVM predicts : ')
    svm = SVC()
    svm.fit(X_train, Y_train)
    predictions = svm.predict(X_validation)
    print(accuracy_score(Y_validation, predictions))
    print(confusion_matrix(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))
    return


def test(i):
    x = i
    X_test = dataset.values[x, 0:4]
    Y_test = dataset.values[x, 4]
    knn = KNeighborsClassifier()
    knn.fit(X_train, Y_train)
    print('test ... actuals : ', x, ' :: ', knn.predict([X_test]), Y_test)


if __name__ == '__main__':
    print('Revving up ...')
    dataset = load_dataset()
    print('Loaded ...')
    # check_dataset()
    # visualize_dataset()
    print('Building models ...')
    X_train, X_validation, Y_train, Y_validation = create_validation_dataset()
    build_models(compare_algorithms=False)
    print('Now, let us see ...')
    make_predictions_knn()
    make_predictions_SVM()
    print('Sample test ...')
    for i in range(dataset.shape[0]):
        test(i)
