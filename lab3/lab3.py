#!/usr/bin/env python3
import sys

import numpy as np
import pandas as pd
# STUDENT SHALL ADD NEEDED IMPORTS

from common import describe_data, test_env, classification_metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import ComplementNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def read_data(file):
    """Return pandas dataFrame read from Excel file"""
    try:
        return pd.read_excel(file)
    except FileNotFoundError:
        sys.exit('ERROR: ' + file + ' not found')


def preprocess_data(df, verbose=False):
    y_column = 'In university after 4 semesters'

    # Features can be excluded by adding column name to list
    drop_columns = []

    categorical_columns = [
        'Faculty',
        'Paid tuition',
        'Study load',
        'Previous school level',
        'Previous school study language',
        'Recognition',
        'Study language',
        'Foreign student'
    ]

    # Handle dependent variable
    if verbose:
        print('Missing y values: ', df[y_column].isna().sum())

    y = df[y_column].values
    # Encode y. Naive solution
    y = np.where(y == 'No', 0, y)
    y = np.where(y == 'Yes', 1, y)
    y = y.astype(float)

    # Drop also dependent variable variable column to leave only features
    drop_columns.append(y_column)
    df = df.drop(labels=drop_columns, axis=1)

    # Remove drop columns for categorical columns just in case
    categorical_columns = [
        i for i in categorical_columns if i not in drop_columns]

    # STUDENT SHALL ENCODE CATEGORICAL FEATURES
    # Include all categorical features columns since they are all objects
    for column in categorical_columns:
        df[column] = df[column].fillna(value='Missing')
    for column in categorical_columns:
        df = pd.get_dummies(df, prefix=[column], columns=[
            column], drop_first=True)

    # Handle missing data. At this point only exam points should be missing
    # It seems to be easier to fill whole data frame as only particular columns
    if verbose:
        describe_data.print_nan_counts(df)

    # STUDENT SHALL HANDLE MISSING VALUES
    df = df.fillna(0)
    # return also the independent variables ready to train
    X = df.astype(float)

    if verbose:
        describe_data.print_nan_counts(df)

    # Return features data frame and dependent variable scaled and
    # ready to train
    return X, y


# STUDENT SHALL CREATE FUNCTIONS FOR LOGISTIC REGRESSION CLASSIFIER, KNN
# CLASSIFIER, SVM CLASSIFIER, NAIVE BAYES CLASSIFIER, DECISION TREE
# CLASSIFIER AND RANDOM FOREST CLASSIFIER
def logistic_regression(X, y):
    # Split test and training data (0.25 is default). Scale features.
    # Train regressor. Predict
    # test set results and print out metrics
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Using sag because it is advisable for large data sets.
    clf = LogisticRegression(random_state=0, solver='sag')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_true = y_test
    label = "Logistic regression test data"

    classification_metrics.print_metrics(y_true, y_pred, label, verbose=True)


def k_nn(X, y):
    # Split test and training data (0.25 is default). Scale features.
    # Train regressor.
    # test set results and print out metrics
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # By default 5 neighbours. I increase neighbours
    # and include Euclidean distance.
    clf = KNeighborsClassifier(n_neighbors=10, p=2)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_true = y_test
    label = "KNN test data"

    classification_metrics.print_metrics(y_true, y_pred, label, verbose=True)


def svc_clasiffier(X, y):
    # Split test and training data (0.25 is default). Scale features. Train regressor. Predict
    # test set results and print out metrics

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Using sigmoid, small gamma and bigger tolerations as per recommendation because
    # it is supposed to be the right parameters for our data model and enabling probability estimation.
    clf = SVC(kernel='sigmoid', gamma=0.3, tol=1e-1, probability=True)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_true = y_test
    label = "SVC test data"

    classification_metrics.print_metrics(y_true, y_pred, label, verbose=True)


def naive_bayes(X, y):
    # Split test and training data (0.25 is default). Scale features. Train regressor. Predict
    # test set results and print out metrics

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0)

    # use min max scaler to overcome the constrain of negative numbers with ComplementNB
    sc = MinMaxScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Using ComplementNB
    clf = ComplementNB()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_true = y_test
    label = "Naive bayes test data"

    classification_metrics.print_metrics(y_true, y_pred, label, verbose=True)


def decision_tree_class(X, y):
    # Split test and training data (0.25 is default). Scale features. Train regressor. Predict
    # test set results and print out metrics

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0)

    # Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_true = y_test
    label = "Decision tree classifier test data"

    classification_metrics.print_metrics(y_true, y_pred, label, verbose=True)


def random_forest(X, y):
    # Split test and training data (0.25 is default). Scale features. Train regressor. Predict
    # test set results and print out metrics

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0)

    # Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # We use 12 estimators.
    clf = RandomForestClassifier(n_estimators=12, random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_true = y_test
    label = "Random forest classifier test data"

    classification_metrics.print_metrics(y_true, y_pred, label, verbose=True)


if __name__ == '__main__':
    modules = ['numpy', 'pandas', 'sklearn']
    test_env.versions(modules)

    students = read_data('data/students.xlsx')
    # STUDENT SHALL CALL PRINT_OVERVIEW AND PRINT_CATEGORICAL FUNCTIONS WITH
    # FILE NAME AS ARGUMENT
    # def print_overview(data_frame, file='')
    describe_data.print_overview(
        students, file='results/2017_students_overview.txt')
    describe_data.print_categorical(students, columns=['Faculty',
                                                       'Paid tuition',
                                                       'Study load',
                                                       'Previous school level',
                                                       'Previous school study language',
                                                       'Recognition',
                                                       'Study language',
                                                       'Foreign student'],
                                    file='results/2017_students_categorical_features.txt')

    # Filter out students not continuing studies (In university aſter 4 semesters is No)
    studentsNotContinuing = students[students['In university after 4 semesters'] == 'No']
    describe_data.print_overview(
        studentsNotContinuing, file='results/2017_studentsNotContinuing_overview.txt')
    describe_data.print_categorical(studentsNotContinuing, columns=['Faculty',
                                                                    'Paid tuition',
                                                                    'Study load',
                                                                    'Previous school level',
                                                                    'Previous school study language',
                                                                    'Recognition',
                                                                    'Study language',
                                                                    'Foreign student'],
                                    file='results/2017_studentsNotContinuing_categorical_features.txt')

    students_X, students_y = preprocess_data(students)
    # STUDENT SHALL CALL CREATED CLASSIFIERS FUNCTIONS
    logistic_regression(students_X, students_y)
    k_nn(students_X, students_y)
    svc_clasiffier(students_X, students_y)
    naive_bayes(students_X, students_y)
    decision_tree_class(students_X, students_y)
    random_forest(students_X, students_y)
    print('Done')
