import pandas as pd
import numpy as np
import math, sys
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error

plt.rc('font', size=18); plt.rcParams['figure.constrained_layout.use'] = True
# read data. column 1 is date/time, col 6 is #bikes
df = pd.read_csv("bikesdata.csv", usecols = [0,1,3,4,6], parse_dates=[1])
kilmainham = df[df['STATION ID'] == 97].copy()
charlemont = df[df['STATION ID'] == 5].copy()


start=pd.to_datetime("27-01-2020",format='%d-%m-%Y')
end=pd.to_datetime("14-03-2020",format='%d-%m-%Y')
kilmainham = kilmainham.loc[(kilmainham['TIME'] > start) & (kilmainham['TIME'] <= end)].copy()

kilmainham_timestamps = kilmainham["TIME"].tolist()
kilmainham_occupancy = kilmainham["AVAILABLE BIKES"].tolist()
#plt.scatter(kilmainham_timestamps, kilmainham_occupancy, color='black')
#plt.show()
charlemont = charlemont.loc[(charlemont['TIME'] > start) & (charlemont['TIME'] <= end)].copy()

charlemont_timestamps = charlemont["TIME"].tolist()
charlemont_occupancy = charlemont["AVAILABLE BIKES"].tolist()
#plt.scatter(charlemont_timestamps, charlemont_occupancy, color='black')
#plt.show()



def test_pred(timestamps, occupancy, q,d,n, plot, station):
    # num_features*d = max offset from k-q => k-q-d*n
    # => k = i+q+d*n
    i = 0
    features = []
    targets = []
    time_domain = []
    while(i+q+(d*n) < len(occupancy)):
        feature_vector = []
        for index in range(n):
            feature_vector.append(occupancy[i+(index*d)])
        features.append(feature_vector)
        targets.append(occupancy[i+q+(d*n)])
        time_domain.append(timestamps[i+(d*n)])
        i += 1
    accuracies = [0,0,0,0,0,0]
    for train, test in KFold(n_splits=5).split(features):
        knn_model = KNeighborsRegressor(n_neighbors=n,weights='uniform').fit(np.array(features)[train], np.array(targets)[train])
        lr_model = LinearRegression(fit_intercept=False).fit(np.array(features)[train], np.array(targets)[train])
        dummy_model = DummyRegressor(strategy="mean").fit(np.array(features)[train], np.array(targets)[train])
        dummy_preds = dummy_model.predict(np.array(features)[test])
        knn_preds = knn_model.predict(np.array(features)[test])
        lr_preds = lr_model.predict(np.array(features)[test])
        knn_mse_accuracy = mean_squared_error(np.array(targets)[test],knn_preds)
        lr_mse_accuracy = mean_squared_error(np.array(targets)[test],lr_preds)
        dummy_mse_accuracy = mean_squared_error(np.array(targets)[test],dummy_preds)
        
        knn_mae_accuracy = mean_absolute_error(np.array(targets)[test],knn_preds)
        lr_mae_accuracy = mean_absolute_error(np.array(targets)[test],lr_preds)
        dummy_mae_accuracy = mean_absolute_error(np.array(targets)[test],dummy_preds)
        
        # Store MSE + MAE accuracies
        accuracies[0] += knn_mse_accuracy
        accuracies[1] += lr_mse_accuracy
        accuracies[2] += dummy_mse_accuracy
        accuracies[3] += knn_mae_accuracy
        accuracies[4] += lr_mae_accuracy
        accuracies[5] += dummy_mae_accuracy


        if plot:
            plt.scatter(timestamps, occupancy, color='black')
            plt.scatter(np.array(time_domain)[test], knn_preds, color='blue')
            plt.title("Q = {}, N = {}".format(q,n))
            plt.xlabel("Time")
            plt.ylabel("Bike Station Occupancy")
            plt.legend(["training data", "predictions"], loc="upper right")
            plt.show()
            plt.cla()
    knn_model = KNeighborsRegressor(n_neighbors=n,weights='uniform').fit(np.array(features), np.array(targets))

    # Get averages of accuracies from the 5 folds
    accuracies = np.array(accuracies)/5
    print("\n\n{}, q = {}, n = {}\nkNN Model MSE = {}, Linear Regression MSE = {}, Dummy Regressor MSE - {}\nkNN Model MAE = {}, Linear Regression MAE = {}, Dummy Regressor MAE - {} ".format(station,q,n, round(accuracies[0],2), round(accuracies[1],2), round(accuracies[2],2),round(accuracies[3],2), round(accuracies[4],2), round(accuracies[5],2)))
    return accuracies

def plot_accuracies(accuracies, window, d):
    knn_mse_accuracies = [x[0] for x in accuracies]
    lr_mse_accuracies = [x[1] for x in accuracies]
    dummy_mse_accuracies = [x[2] for x in accuracies]
    knn_mae_accuracies = [x[3] for x in accuracies]
    lr_mae_accuracies = [x[4] for x in accuracies]
    dummy_mae_accuracies = [x[5] for x in accuracies]

    plt.subplot(1,2,1)
    plt.scatter(window,knn_mse_accuracies, color="blue")
    plt.scatter(window,lr_mse_accuracies, color="green")
    plt.scatter(window,dummy_mse_accuracies, color="red")
    plt.title("Mean Squared Error")
    plt.xlabel("Feature Vector Size")
    plt.ylabel("accuracy")
    plt.legend(["kNN Model", "Linear Regression", "Baseline"], loc="upper right", prop={'size': 6})
    plt.yscale("log")
    plt.yticks([5,10,20,50,100,150])

    plt.subplot(1,2,2)
    plt.scatter(window,knn_mae_accuracies, color="blue")
    plt.scatter(window,lr_mae_accuracies, color="green")
    plt.scatter(window,dummy_mae_accuracies, color="red")
    plt.title("Mean Absolute Error")
    plt.xlabel("Feature Vector Size")
    plt.ylabel("accuracy")
    plt.legend(["kNN Model", "Linear Regression", "Baseline"], loc="upper right", prop={'size': 6})
    plt.yscale("log")
    plt.yticks([5,10,20,50,100,150])


    plt.show()
# -- KILMAINHAM --
"""
# 10 minutes ahead - q = 2, using last 5 values
window = [3,4,5,6,7,8,9, 15,30]
accuracies = []
for n in window:
    accuracies.append(test_pred(kilmainham_timestamps, kilmainham_occupancy, 2,1,n,False, "Kilmainham"))

# plot_accuracies(accuracies, window,1)

# 30 minutes ahead - q = 6, using last 7 values
window = [2,3,4,5,6,7,8,9, 15,30]
accuracies = []
for n in window:
    accuracies.append(test_pred(kilmainham_timestamps, kilmainham_occupancy, 6,1,n,False, "Kilmainham"))

plot_accuracies(accuracies, window,1)

# 1 hour ahead - q = 10, using short term data
window = [2,3,4,5,6,7,8,9,15,30]
accuracies = []
for n in window:
    accuracies.append(test_pred(kilmainham_timestamps, kilmainham_occupancy, 10,1,n,False, "Kilmainham"))

plot_accuracies(accuracies, window,1)



# 1 hour ahead - q = 10, using daily data
window = [2,3,4,5]
accuracies = []
for n in window:
    accuracies.append(test_pred(kilmainham_timestamps, kilmainham_occupancy, 10,288,n,False, "Kilmainham"))

plot_accuracies(accuracies, window, 288)



# 1 hour ahead - q = 10, using weekly data
window = [2,3,4,5]
accuracies = []
for n in window:
    accuracies.append(test_pred(kilmainham_timestamps, kilmainham_occupancy, 10,2016,n,False, "Kilmainham"))

plot_accuracies(accuracies, window, 288)



# -- CHARLEMONT PLACE -- 

# 10 minutes ahead - q = 2, using last 5 values
window = [3,4,5,6,7,8,9, 15,30]
accuracies = []
for n in window:
    accuracies.append(test_pred(charlemont_timestamps, charlemont_occupancy, 2,1,n,True, "Charlemont"))

plot_accuracies(accuracies, window,1)

# 30 minutes ahead - q = 6, using last 7 values
window = [2,3,4,5,6,7,8,9, 15,30]
accuracies = []
for n in window:
    accuracies.append(test_pred(charlemont_timestamps, charlemont_occupancy, 6,1,n,False, "Charlemont"))

plot_accuracies(accuracies, window,1)

# 1 hour ahead - q = 10, using short term data
window = [2,3,4,5,6,7,8,9,15,30]
accuracies = []
for n in window:
    accuracies.append(test_pred(charlemont_timestamps, charlemont_occupancy, 10,1,n,False, "Charlemont"))

plot_accuracies(accuracies, window,1)




# 1 hour ahead - q = 10, using daily data
window = [2,3,4,5]
accuracies = []
for n in window:
    accuracies.append(test_pred(charlemont_timestamps, charlemont_occupancy, 10,288,n,False, "Charlemont"))

plot_accuracies(accuracies, window, 288)

"""
# 1 hour ahead - q = 10, using weekly data
window = [2,3,4,5]
accuracies = []
for n in window:
    accuracies.append(test_pred(charlemont_timestamps, charlemont_occupancy, 10,2016,n,False, "Charlemont"))

plot_accuracies(accuracies, window, 288)
