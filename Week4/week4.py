import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

# -- FORMAT DATASET --
data = open("data","r").read().split("\n")
dataset_id_one = data[0]
data = data[1:]
data = data[:-1]

index = 0
for row in data:
    if row[0] == "#":
        break
    index += 1

data_one = data[:index]
data_two = data[index:]
dataset_id_two = data_two[0]
data_two = data_two[1:]

print("DATASET ONE ID: {}".format(dataset_id_one))
print("DATASET TWO ID: {}".format(dataset_id_two))


data_one = [x.split(",") for x in data_one]
data_two = [x.split(",") for x in data_two]

f1_one = []
f2_one = []
f1_two = []
f2_two = []
features_one = []
features_two = []
label_one = []
label_two = []

for (row_one, row_two) in zip(data_one,data_two):
    f1_one.append(float(row_one[0]))
    f2_one.append(float(row_one[1]))
    features_one.append([float(row_one[0]), float(row_one[1])])
    label_one.append(float(row_one[2]))

    f1_two.append(float(row_two[0]))
    f2_two.append(float(row_two[1]))
    features_two.append([float(row_two[0]), float(row_two[1])])
    label_two.append(float(row_two[2]))

# -- END OF FORMATTING

# --- FIRST DATASET ---

# -- QUESTION ONE --
# a
# Acquire the accuracy and predictions of each Logistic Regression model configured with C and polynomial orders.


C_values = [0.01, 0.1, 1, 10, 100]
poly_orders = [1,2,3,4,5,6]
dataframe_values = []
all_accuracy_stds = []
all_accuracies_for_plots = []


colormap = ListedColormap(["r", "g"])
pred_colormap = ListedColormap(["b","purple"])

classes = ["Target With Label -1", "Target with Label 1"]
selected_logreg_model = None
selected_logreg_model_accuracy = 0
selected_lr_poly_order = 1


for c in C_values:
    accuracy_stds = []
    accuracy_for_plot = []
    for p in poly_orders:
        split_vals = []
        predictions = []
        plotted = False
        for train, test in KFold(n_splits=5).split(features_one):
            x_poly = PolynomialFeatures(p).fit_transform(np.array(features_one)[train])
            x_poly_test = PolynomialFeatures(p).fit_transform(np.array(features_one)[test])
            model = LogisticRegression(penalty='l2',C=c, max_iter=500).fit(x_poly,np.array(label_one)[train])
            prediction = model.predict(x_poly_test)
            predictions.append(prediction)
            accuracy = accuracy_score(np.array(label_one)[test], prediction, normalize=True)
            split_vals.append([c,p,accuracy])
            if (c==0.01 or c==1 or c == 100) and not plotted:
                # Plot each prediction
                categories = [x if x != -1 else 0 for x in np.array(label_one)[train]]
                plot =  plt.scatter(np.array(f1_one)[train], np.array(f2_one)[train], c=categories, cmap=colormap)
                plt.title("Accuracy: {}\nC-Value: {}\nOrder: {}".format(accuracy,c,p))
                plt.xlabel("First feature")
                plt.ylabel("Second feature")
                plt.legend(handles=plot.legend_elements()[0], labels=classes)
                pred_categories = [x if x != -1 else 0 for x in prediction]
                pred_plot = plt.scatter(np.array(f1_one)[test],np.array(f2_one)[test], c=pred_categories, cmap=pred_colormap, marker="x")
                # plt.show()
            plotted=True
        accuracy_stds.append(np.array([x[2] for x in split_vals]).std())
        split_vals = np.array(split_vals).mean(axis=0)
        accuracy_for_plot.append(split_vals[2])
        dataframe_values.append(split_vals)

        # Update model to be the one with the greatest accuracy
        if selected_logreg_model == None or split_vals[2] > selected_logreg_model_accuracy:
            selected_logreg_model = model
            selected_logreg_model_accuracy = split_vals[2]
            selected_lr_poly_order = p
    all_accuracy_stds.append(accuracy_stds)
    all_accuracies_for_plots.append(accuracy_for_plot)


# -- Plot the accuracies of each config and their standard deviations for Logistic Regression --

subplot_index = 1
c_index = 0


for accuracy,std in zip(all_accuracies_for_plots,all_accuracy_stds):
    plt.subplot(3,2,subplot_index)
    plt.grid()
    plt.title("C-Value - {}".format(C_values[c_index]))
    plt.errorbar(poly_orders,accuracy,std)
    subplot_index += 1
    c_index += 1

# plt.show()
print("Accuracy of selected Logistic Regression Model: {}".format(selected_logreg_model_accuracy))
print(selected_logreg_model)
# -- End of plotting accuracies

# b
# kNN 
selected_knn_model = None
selected_knn_model_accuracy = 0
k_values = [3,4,5,6,7,8,9,10]
subplot_index = 1
for k in k_values:
    accuracies = []
    plotted = False
    for train, test in KFold(n_splits=5).split(features_one):
        model = KNeighborsClassifier(n_neighbors=k,weights="uniform").fit(np.array(features_one)[train], np.array(label_one)[train])
        prediction = model.predict(np.array(features_one)[test])
        accuracy = accuracy_score(np.array(label_one)[test], prediction, normalize=True)
        accuracies.append(accuracy)
        if not plotted:
            categories = [x if x != -1 else 0 for x in np.array(label_one)[train]]
            plot =  plt.scatter(np.array(f1_one)[train], np.array(f2_one)[train], c=categories, cmap=colormap)
            plt.title("Accuracy: {}\nK-value: {}".format(accuracy,k))
            plt.xlabel("First feature")
            plt.ylabel("Second feature")
            plt.legend(handles=plot.legend_elements()[0], labels=classes)
            pred_categories = [x if x != -1 else 0 for x in prediction]
            pred_plot = plt.scatter(np.array(f1_one)[test],np.array(f2_one)[test], c=pred_categories, cmap=pred_colormap, marker="x")
            # plt.show()
        plotted = True
    accuracy = np.array(accuracies).mean()   
    if selected_knn_model == None or accuracy > selected_knn_model_accuracy:
            selected_knn_model = model
            selected_knn_model_accuracy = accuracy
    print("K-values: {} Accuracy: {}".format(k,np.array(accuracies).mean()))

print("Accuracy of knn model: {}".format(selected_knn_model_accuracy))
print(selected_knn_model)

# c
# -- Calculate confusion matrix for both the kNN classifier and regression classifier

train_features,test_features,train_labels,test_labels = train_test_split(features_one,label_one,test_size=0.25,random_state=0)
knn_preds = selected_knn_model.predict(test_features)
lr_preds = selected_logreg_model.predict(PolynomialFeatures(selected_lr_poly_order).fit_transform(test_features))
dummy_model = DummyClassifier(strategy="most_frequent").fit(train_features,train_labels)
dummy_preds = dummy_model.predict(test_features)
knn_tn, knn_fp, knn_fn, knn_tp = confusion_matrix(test_labels, knn_preds).ravel()
lr_tn, lr_fp, lr_fn, lr_tp = confusion_matrix(test_labels, lr_preds).ravel()
bc_tn, bc_fp, bc_fn, bc_tp = confusion_matrix(test_labels, dummy_preds).ravel()


print("\n\nkNN Classifier Confusion Matrix\n\nTrue Negative: {}\nFalse Postive: {}\nFalse Negative: {}\nTrue Positive: {}".format(knn_tn, knn_fp, knn_fn, knn_tp))
print(" \n-----\nLogistic Regression Confusion Matrix\n\nTrue Negative: {}\nFalse Postive: {}\nFalse Negative: {}\nTrue Positive: {}".format(lr_tn, lr_fp, lr_fn, lr_tp))
print(" \n-----\nMost Frequent Label Classifier Confusion Matrix\n\nTrue Negative: {}\nFalse Postive: {}\nFalse Negative: {}\nTrue Positive: {}".format(bc_tn, bc_fp, bc_fn, bc_tp))

# d 
# Calculate and plot roc curves for knn classifier, lr model and dummy classifier

plt.cla()
plt.clf()

fpr, tpr, _ = roc_curve(test_labels,knn_preds)

plt.plot(fpr,tpr)

fpr, tpr, _ = roc_curve(test_labels, selected_logreg_model.decision_function(PolynomialFeatures(selected_lr_poly_order).fit_transform(test_features)))
plt.plot(fpr,tpr,color="orange")

fpr, tpr, _ = roc_curve(test_labels,dummy_preds)

plt.plot(fpr,tpr, color="green")
plt.legend(["KNN Classifier", "Logistic Regression", "Baseline Classifier"])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
# plt.show()

# --- SECOND DATASET ---

# -- QUESTION ONE --
# a
# Acquire the accuracy and predictions of each Logistic Regression model configured with C and polynomial orders.
plt.cla()
plt.clf()

C_values = [0.01, 0.1, 1, 10, 100]
poly_orders = [1,2,3,4,5,6]
dataframe_values = []
all_accuracy_stds = []
all_accuracies_for_plots = []


colormap = ListedColormap(["r", "g"])
pred_colormap = ListedColormap(["b","purple"])

classes = ["Target With Label -1", "Target with Label 1"]
selected_logreg_model = None
selected_logreg_model_accuracy = 0
selected_lr_poly_order = 1


for c in C_values:
    accuracy_stds = []
    accuracy_for_plot = []
    for p in poly_orders:
        split_vals = []
        predictions = []
        plotted = False
        for train, test in KFold(n_splits=5).split(features_two):
            x_poly = PolynomialFeatures(p).fit_transform(np.array(features_two)[train])
            x_poly_test = PolynomialFeatures(p).fit_transform(np.array(features_two)[test])
            model = LogisticRegression(penalty='l2',C=c, max_iter=500).fit(x_poly,np.array(label_two)[train])
            prediction = model.predict(x_poly_test)
            predictions.append(prediction)
            accuracy = accuracy_score(np.array(label_two)[test], prediction, normalize=True)
            split_vals.append([c,p,accuracy])
            if (c==0.01 or c==1 or c == 100) and not plotted:
                # Plot each prediction
                categories = [x if x != -1 else 0 for x in np.array(label_two)[train]]
                plot =  plt.scatter(np.array(f1_two)[train], np.array(f2_two)[train], c=categories, cmap=colormap)
                plt.title("Accuracy: {}\nC-Value: {}\nOrder: {}".format(accuracy,c,p))
                plt.xlabel("First feature")
                plt.ylabel("Second feature")
                plt.legend(handles=plot.legend_elements()[0], labels=classes)
                pred_categories = [x if x != -1 else 0 for x in prediction]
                pred_plot = plt.scatter(np.array(f1_two)[test],np.array(f2_two)[test], c=pred_categories, cmap=pred_colormap, marker="x")
                plt.show()
            plotted=True
        accuracy_stds.append(np.array([x[2] for x in split_vals]).std())
        split_vals = np.array(split_vals).mean(axis=0)
        accuracy_for_plot.append(split_vals[2])
        dataframe_values.append(split_vals)

        # Update model to be the one with the greatest accuracy
        if selected_logreg_model == None or split_vals[2] > selected_logreg_model_accuracy:
            selected_logreg_model = model
            selected_logreg_model_accuracy = split_vals[2]
            selected_lr_poly_order = p
    all_accuracy_stds.append(accuracy_stds)
    all_accuracies_for_plots.append(accuracy_for_plot)


# -- Plot the accuracies of each config and their standard deviations for Logistic Regression --

subplot_index = 1
c_index = 0
print(len(all_accuracy_stds))
print(len(all_accuracies_for_plots))
for accuracy,std in zip(all_accuracies_for_plots,all_accuracy_stds):
    plt.subplot(3,2,subplot_index)
    plt.grid()
    plt.title("C-Value - {}".format(C_values[c_index]))
    plt.errorbar(poly_orders,accuracy,std)
    subplot_index += 1
    c_index += 1

plt.show()
print("Accuracy of selected Logistic Regression Model: {}".format(selected_logreg_model_accuracy))
# -- End of plotting accuracies

# b
# kNN 
selected_knn_model = None
selected_knn_model_accuracy = 0
k_values = [3,4,5,6,7,8,9,10]
subplot_index = 1
for k in k_values:
    accuracies = []
    plotted = False
    for train, test in KFold(n_splits=5).split(features_two):
        model = KNeighborsClassifier(n_neighbors=k,weights="uniform").fit(np.array(features_two)[train], np.array(label_one)[train])
        prediction = model.predict(np.array(features_two)[test])
        accuracy = accuracy_score(np.array(label_two)[test], prediction, normalize=True)
        accuracies.append(accuracy)
        if not plotted:
            categories = [x if x != -1 else 0 for x in np.array(label_two)[train]]
            plot =  plt.scatter(np.array(f1_two)[train], np.array(f2_two)[train], c=categories, cmap=colormap)
            plt.title("Accuracy: {}\nK-value: {}".format(accuracy,k))
            plt.xlabel("First feature")
            plt.ylabel("Second feature")
            plt.legend(handles=plot.legend_elements()[0], labels=classes)
            pred_categories = [x if x != -1 else 0 for x in prediction]
            pred_plot = plt.scatter(np.array(f1_two)[test],np.array(f2_two)[test], c=pred_categories, cmap=pred_colormap, marker="x")
            plt.show()
        plotted = True
    accuracy = np.array(accuracies).mean()   
    if selected_knn_model == None or accuracy > selected_knn_model_accuracy:
            selected_knn_model = model
            selected_knn_model_accuracy = accuracy
    print("K-values: {} Accuracy: {}".format(k,np.array(accuracies).mean()))

print("Accuracy of knn model: {}".format(selected_knn_model_accuracy))

# c
# -- Calculate confusion matrix for both the kNN classifier and regression classifier

train_features,test_features,train_labels,test_labels = train_test_split(features_two,label_two,test_size=0.25,random_state=0)
knn_preds = selected_knn_model.predict(test_features)
lr_preds = selected_logreg_model.predict(PolynomialFeatures(selected_lr_poly_order).fit_transform(test_features))
dummy_model = DummyClassifier(strategy="most_frequent").fit(train_features,train_labels)
dummy_preds = dummy_model.predict(test_features)
knn_tn, knn_fp, knn_fn, knn_tp = confusion_matrix(test_labels, knn_preds).ravel()
lr_tn, lr_fp, lr_fn, lr_tp = confusion_matrix(test_labels, lr_preds).ravel()
bc_tn, bc_fp, bc_fn, bc_tp = confusion_matrix(test_labels, dummy_preds).ravel()


print("\n\nkNN Classifier Confusion Matrix\n\nTrue Negative: {}\nFalse Postive: {}\nFalse Negative: {}\nTrue Positive: {}".format(knn_tn, knn_fp, knn_fn, knn_tp))
print(" \n-----\nLogistic Regression Confusion Matrix\n\nTrue Negative: {}\nFalse Postive: {}\nFalse Negative: {}\nTrue Positive: {}".format(lr_tn, lr_fp, lr_fn, lr_tp))
print(" \n-----\nMost Frequent Label Classifier Confusion Matrix\n\nTrue Negative: {}\nFalse Postive: {}\nFalse Negative: {}\nTrue Positive: {}".format(bc_tn, bc_fp, bc_fn, bc_tp))

# d 
# Calculate and plot roc curves for knn classifier, lr model and dummy classifier

plt.cla()
plt.clf()

fpr, tpr, _ = roc_curve(test_labels,knn_preds)

plt.plot(fpr,tpr)

fpr, tpr, _ = roc_curve(test_labels, selected_logreg_model.decision_function(PolynomialFeatures(selected_lr_poly_order).fit_transform(test_features)))
plt.plot(fpr,tpr,color="orange")

fpr, tpr, _ = roc_curve(test_labels,dummy_preds)

plt.plot(fpr,tpr, color="green")
plt.legend(["KNN Classifier", "Logistic Regression", "Baseline Classifier"])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()