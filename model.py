import pandas as pd
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, mean_squared_error, r2_score
from numpy import *
import seaborn as sns  
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def get_data(filepath):
    data_in = genfromtxt(filepath, delimiter=",")

    # Data preprocessing (onehot encoding)
    gender = np.copy(data_in[:, 0])
    onehot_encoder = OneHotEncoder(sparse=False)
    gender = gender.reshape(len(gender), 1)
    onehot_encoded = onehot_encoder.fit_transform(gender)
    data_in = np.hstack((onehot_encoded, data_in[:,1:]))

    # Separate feature x and target y
    data_x = data_in[:,:-1]
    data_y = data_in[:,-1]

    return data_x, data_y


def correlation_matrix(data_x, data_y):
    # Normalize features and combine with ring age
    scaler = MinMaxScaler().fit(data_x)
    data_x = scaler.transform(data_x)
    data_y = np.expand_dims(data_y, axis=1)
    data_in = np.hstack((data_x, data_y))

    # Generate Correlation Matrix
    corr_mat = np.corrcoef(data_in.T)
    sns.heatmap(data=corr_mat, annot=True)
    plt.title('Correlation Matrix')
    plt.xlabel("Feature")
    plt.ylabel("Feature")
    plt.savefig('heatmap.png')
    plt.clf()


# Generate Scatter Plot
def scatterplot(data_x, data_y, feature_name_x, feature_name_y):
    plt.scatter(data_x, data_y)
    plt.title(feature_name_x + ' - ' + feature_name_y +' Scatter Plot')
    plt.xlabel(feature_name_x)
    plt.ylabel(feature_name_y)
    plt.savefig(feature_name_x + '_' + feature_name_y +'_scatterplot.png')
    plt.clf()


# Generate Histogram
def histogram(data, feature_name):
    plt.hist(data, bins='auto')
    plt.title(feature_name+' Histogram')    
    plt.xlabel(feature_name)
    plt.ylabel('Count')
    plt.savefig(feature_name + '_histogram.png')
    plt.clf()


def confidence_interval(data, alpha):
    return st.t.interval(confidence=alpha, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))


def plot_acc(df_train, df_test, plotname, xname):
    fig, ax = plt.subplots()
    x = df_train[xname]
    ax.plot(x, df_train['mean'], color='b', label='Train')
    ax.plot(x, df_test['mean'], color='r', label='Test')
    ax.fill_between(
        x, df_train['ci_lower'], df_train['ci_upper'], color='b', alpha=.15)
    ax.fill_between(
        x, df_test['ci_lower'], df_test['ci_upper'], color='r', alpha=.15)
    x_list = df_train[xname].tolist()
    ax.set_xticks(x_list)
    ax.set_xlabel(xname)
    # ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    ax.set_title(plotname)
    plt.legend(loc="upper left")
    plt.savefig(f'{plotname}.png')
    plt.clf()


def plot_bar(df_train, df_test, plotname, xname):
    plt.figure(figsize=(6.4, 4.8))
    bar_names = df_train[xname].to_list()
    ind = np.arange(len(bar_names))
    width = 0.3
    plt.bar(ind, df_train['mean'], width, color='b', label='Train')
    plt.bar(ind+width, df_test['mean'], width, color='r', label='Test')
    plt.errorbar(ind, df_train['mean'], yerr=df_train['error'], fmt='o', color='Black', elinewidth=2, capthick=2, errorevery=1,
                 alpha=1, ms=4, capsize=18)
    plt.errorbar(ind+width, df_test['mean'], yerr=df_test['error'], fmt='o', color='Black', elinewidth=2, capthick=2, errorevery=1,
                 alpha=1, ms=4, capsize=18)
    plt.xlabel(xname)
    plt.title(plotname)
    plt.xticks(ind + width / 2, bar_names)
    plt.legend(loc='best')
    plt.savefig(f'{plotname}.png')
    plt.clf()


# https://github.com/vinyluis
def calculate_tpr_fpr(y_real, y_pred):
    '''
    Calculates the True Positive Rate (tpr) and the True Negative Rate (fpr) based on real and predicted observations

    Args:
        y_real: The list or series with the real classes
        y_pred: The list or series with the predicted classes

    Returns:
        tpr: The True Positive Rate of the classifier
        fpr: The False Positive Rate of the classifier
    '''

    # Calculates the confusion matrix and recover each element
    cm = confusion_matrix(y_real, y_pred)
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TP = cm[1, 1]

    # Calculates tpr and fpr
    tpr = TP / (TP + FN)  # sensitivity - true positive rate
    fpr = 1 - TN / (TN + FP)  # 1-specificity - false positive rate

    return tpr, fpr


# https://github.com/vinyluis
def get_all_roc_coordinates(y_real, y_proba):
    '''
    Calculates all the ROC Curve coordinates (tpr and fpr) by considering each point as a threshold for the predicion of the class.

    Args:
        y_real: The list or series with the real classes.
        y_proba: The array with the probabilities for each class, obtained by using the `.predict_proba()` method.

    Returns:
        tpr_list: The list of TPRs representing each threshold.
        fpr_list: The list of FPRs representing each threshold.
    '''
    tpr_list = [0]
    fpr_list = [0]
    for i in range(len(y_proba)):
        threshold = y_proba[i]
        y_pred = y_proba >= threshold
        tpr, fpr = calculate_tpr_fpr(y_real, y_pred)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    return tpr_list, fpr_list


# https://github.com/vinyluis
def plot_roc_curve(tpr, fpr, scatter=True, ax=None):
    '''
    Plots the ROC Curve by using the list of coordinates (tpr and fpr).

    Args:
        tpr: The list of TPRs representing each coordinate.
        fpr: The list of FPRs representing each coordinate.
        scatter: When True, the points used on the calculation will be plotted with the line (default = True).
    '''
    if ax == None:
        plt.figure(figsize=(5, 5))
        ax = plt.axes()

    if scatter:
        sns.scatterplot(x=fpr, y=tpr, ax=ax)
    sns.lineplot(x=fpr, y=tpr, ax=ax)
    sns.lineplot(x=[0, 1], y=[0, 1], color='green', ax=ax)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")


# https://github.com/vinyluis
def roc_curve(classes, x_test, y_test, y_proba):
    plt.figure(figsize=(16, 8))
    bins = [i / 20 for i in range(20)] + [1]
    roc_auc_ovr = {}
    x_test = pd.DataFrame(x_test, columns=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    for i in range(len(classes)):
        # Gets the class
        c = classes[i]

        # Prepares an auxiliar dataframe to help with the plots
        df_aux = x_test.copy()
        df_aux['class'] = [1 if y == c else 0 for y in y_test]
        df_aux['prob'] = y_proba[:, i]
        df_aux = df_aux.reset_index(drop=True)

        # Plots the probability distribution for the class and the rest
        ax = plt.subplot(2, 4, i + 1)
        sns.histplot(x="prob", data=df_aux, hue='class', color='b', ax=ax, bins=bins)
        ax.set_title(c)
        ax.legend([f"Class: {c}", "Rest"])
        ax.set_xlabel(f"P(x = {c})")

        # Calculates the ROC Coordinates and plots the ROC Curves
        ax_bottom = plt.subplot(2, 4, i + 5)
        tpr, fpr = get_all_roc_coordinates(df_aux['class'], df_aux['prob'])
        plot_roc_curve(tpr, fpr, scatter=False, ax=ax_bottom)
        ax_bottom.set_title("ROC Curve OvR")

        # Calculates the ROC AUC OvR
        roc_auc_ovr[c] = roc_auc_score(df_aux['class'], df_aux['prob'])
    plt.tight_layout()
    plt.savefig('roc_curve.png')
    plt.clf()
    plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]

    # Save ROC AUC Score to csv
    roc_auc_score_list = []
    for i in range(len(roc_auc_ovr)):
        roc_auc_score_list.append(roc_auc_ovr[i+1])
    roc_auc_score_avg = np.mean(roc_auc_score_list)
    roc_auc_score_df = pd.DataFrame({'class': classes, 'score': roc_auc_score_list})
    roc_auc_score_new_row = {'class': 'mean', 'score': roc_auc_score_avg}
    roc_auc_score_df = pd.concat([roc_auc_score_df, pd.DataFrame([roc_auc_score_new_row])], ignore_index=True)
    roc_auc_score_df.to_csv('roc_auc_score_df', index=False)

def nn_reg(data_x, data_y, percent_test, exp_num=None, normalise=None, hidden=(30,), solver='sgd', epoch=200, learn_rate=0.01):
    # Normalise features
    if normalise:
        scaler = MinMaxScaler().fit(data_x)
        data_x = scaler.transform(data_x)

    # Train-Test Split
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=percent_test, random_state=exp_num)

    # Train Model
    nn = MLPRegressor(hidden_layer_sizes=hidden, max_iter=epoch, solver=solver, learning_rate_init=learn_rate)
    nn.fit(x_train, y_train)

    # Train Result (Check for overfitting)
    y_train_pred = nn.predict(x_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    r2_train = r2_score(y_train, y_train_pred)

    # Test Result
    y_pred = nn.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f'rmse, r2: {rmse}, {r2}')

    if exp_num == 0:
        # Plot residuals
        residuals = y_pred - y_test
        # Determine the maximum absolute value in the data
        max_res = max(abs(residuals))
        plt.ylim(-max_res, max_res)
        plt.scatter(y_test, residuals)
        plt.axhline(0, color='black', linewidth=0.5)
        plt.title(f'Neural Network Regression: Residual vs Ring-age')
        plt.xlabel('Ring-age')
        plt.ylabel('Residual')
        plt.savefig('neural_network_residualvsringage.png')
        plt.clf()

        plt.ylim(-max_res, max_res)
        plt.scatter(x_test[:,-1], residuals)
        plt.axhline(0, color='black', linewidth=0.5)
        plt.title(f'Neural Network Regression: Residual vs Shell Weight')
        plt.xlabel('Shell weight')
        plt.ylabel('Residual')
        plt.savefig('neural_network_residualvsshellweight.png')
        plt.clf()

    return rmse, r2, rmse_train, r2_train


def nn_classifier(data_x, data_y, percent_test, exp_num, epoch, normalise, hidden, solver, L2, learn_rate, plot_roc=False):
    # Normalise features
    if normalise:
        scaler = MinMaxScaler().fit(data_x)
        data_x = scaler.transform(data_x)

    # Train-Test Split
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=percent_test, random_state=exp_num)

    # Train Model
    nn = MLPClassifier(hidden_layer_sizes=hidden, random_state=exp_num, max_iter=epoch, solver=solver, alpha=L2, learning_rate_init=learn_rate )
    nn.fit(x_train, y_train)

    # Make predictions using the testing set
    y_pred_test = nn.predict(x_test)
    y_pred_train = nn.predict(x_train)

    acc_test = accuracy_score(y_pred_test, y_test)
    acc_train = accuracy_score(y_pred_train, y_train)
    
    # plot the roc curve for the model 
    if plot_roc:
        cm = confusion_matrix(y_pred_test, y_test)
        y_proba = nn.predict_proba(x_test)
        classes = nn.classes_
        if exp_num == 0:
            roc_curve(classes, x_test, y_test, y_proba)
        print(f'acc_train, acc_test: {acc_train}, {acc_test}')
        return acc_train, acc_test, cm

    return acc_train, acc_test


def main():
    exp = 10
    data_x, data_y = get_data('abalone_class.data')

    # 1.) Data Analysis
    correlation_matrix(data_x, data_y)
    scatterplot(data_x[:,2], data_y, 'Is Infant', 'Ring-age Class')
    scatterplot(data_x[:,9], data_y, 'Shell Weight', 'Ring-age Class')
    histogram(data_x[:,2], 'Is Infant')
    histogram(data_x[:,9], 'Shell Weight')
    histogram(data_y, 'Ring-age Class')

    # 2.) Find optimal no. of hidden nodes
    hidden_nodes_df_train = pd.DataFrame(columns=['hidden_nodes', 'mean', 'ci_lower', 'ci_upper'])
    hidden_nodes_df_test = pd.DataFrame(columns=['hidden_nodes', 'mean', 'ci_lower', 'ci_upper'])

    for node_num in (5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100):
        hidden_nodes_acc_train = []
        hidden_nodes_acc_test = []
        for i in range(exp):
            acc_train, acc_test = nn_classifier(
                data_x, data_y,
                percent_test=0.4,
                exp_num=i,
                epoch=1000,
                normalise=True,
                hidden=(node_num,),
                solver='sgd',
                L2=0,
                learn_rate=0.01,
                plot_roc=False)
            hidden_nodes_acc_train.append(acc_train)
            hidden_nodes_acc_test.append(acc_test)

        # Calculate confidence interval for test and train
        ci_train = confidence_interval(hidden_nodes_acc_train, 0.95)
        ci_test = confidence_interval(hidden_nodes_acc_test, 0.95)

        # Add data to dataframe for plotting later
        new_row_train = {'hidden_nodes':node_num, 'mean':np.mean(hidden_nodes_acc_train), 'ci_lower':ci_train[0], 'ci_upper':ci_train[1]}
        new_row_test = {'hidden_nodes':node_num, 'mean':np.mean(hidden_nodes_acc_test), 'ci_lower':ci_test[0], 'ci_upper':ci_test[1]}
        hidden_nodes_df_train = pd.concat([hidden_nodes_df_train, pd.DataFrame([new_row_train])], ignore_index=True)
        hidden_nodes_df_test = pd.concat([hidden_nodes_df_test, pd.DataFrame([new_row_test])], ignore_index=True)

    # Save to csv
    hidden_nodes_df_train = hidden_nodes_df_train.astype({"hidden_nodes": int})
    hidden_nodes_df_test = hidden_nodes_df_test.astype({"hidden_nodes": int})
    print(hidden_nodes_df_test)
    hidden_nodes_df_train.to_csv('hidden_nodes_df_train.csv', index=False)
    hidden_nodes_df_test.to_csv('hidden_nodes_df_test.csv', index=False)

    # Plot accuracy vs no. of hidden nodes
    plot_acc(hidden_nodes_df_train, hidden_nodes_df_test, 'Accuracy VS No. of Hidden Nodes', 'hidden_nodes')

    # 3.) Find optimum learning rate
    lr_df_train = pd.DataFrame(columns=['learning_rate', 'mean', 'ci_lower', 'ci_upper'])
    lr_df_test = pd.DataFrame(columns=['learning_rate', 'mean', 'ci_lower', 'ci_upper'])

    for lr in (0.001, 0.01, 0.05, 0.1, 0.25, 0.5):
        lr_acc_train = []
        lr_acc_test = []
        for i in range(exp):
            acc_train, acc_test = nn_classifier(
                data_x, data_y,
                percent_test=0.4,
                exp_num=i,
                epoch=1000,
                normalise=True,
                hidden=(90,),
                solver='sgd',
                L2=0,
                learn_rate=lr,
                plot_roc=False)
            lr_acc_train.append(acc_train)
            lr_acc_test.append(acc_test)

        # Calculate confidence interval for test and train
        ci_train = confidence_interval(lr_acc_train, 0.95)
        ci_test = confidence_interval(lr_acc_test, 0.95)

        # Add data to dataframe for plotting later
        new_row_train = {'learning_rate':lr, 'mean':np.mean(lr_acc_train), 'ci_lower':ci_train[0], 'ci_upper':ci_train[1]}
        new_row_test = {'learning_rate':lr, 'mean':np.mean(lr_acc_test), 'ci_lower':ci_test[0], 'ci_upper':ci_test[1]}
        lr_df_train = pd.concat([lr_df_train, pd.DataFrame([new_row_train])], ignore_index=True)
        lr_df_test = pd.concat([lr_df_test, pd.DataFrame([new_row_test])], ignore_index=True)

    # Save to csv
    lr_df_train = lr_df_train.astype({"learning_rate": float})
    lr_df_test = lr_df_test.astype({"learning_rate": float})
    print(lr_df_test)
    lr_df_train.to_csv('lr_df_train.csv', index=False)
    lr_df_test.to_csv('lr_df_test.csv', index=False)

    # Plot accuracy vs learning rate
    plot_acc(lr_df_train, lr_df_test, 'Accuracy VS Learning Rate', 'learning_rate')

    # 4.) Find optimal no. of hidden layers
    hid_layers_df_train = pd.DataFrame(columns=['hidden_layers', 'mean', 'ci_lower', 'ci_upper'])
    hid_layers_df_test = pd.DataFrame(columns=['hidden_layers', 'mean', 'ci_lower', 'ci_upper'])
    hid_layers = [(90,), (90, 90), (90, 90, 90), (90, 90, 90, 90), (90, 90, 90, 90, 90),
                  (90, 90, 90, 90, 90, 90), (90, 90, 90, 90, 90, 90, 90), (90, 90, 90, 90, 90, 90, 90, 90),
                  (90, 90, 90, 90, 90, 90, 90, 90, 90), (90, 90, 90, 90, 90, 90, 90, 90, 90, 90)]
    for layer_num in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10):
        hid_layers_acc_train = []
        hid_layers_acc_test = []
        for i in range(exp):
            acc_train, acc_test = nn_classifier(
                data_x, data_y,
                percent_test=0.4,
                exp_num=i,
                epoch=1000,
                normalise=True,
                hidden=hid_layers[layer_num-1],
                solver='sgd',
                L2=0,
                learn_rate=0.05,
                plot_roc=False)
            hid_layers_acc_train.append(acc_train)
            hid_layers_acc_test.append(acc_test)

        # Calculate confidence interval for test and train
        ci_train = confidence_interval(hid_layers_acc_train, 0.95)
        ci_test = confidence_interval(hid_layers_acc_test, 0.95)

        # Add data to dataframe for plotting later
        new_row_train = {'hidden_layers': layer_num, 'mean': np.mean(hid_layers_acc_train), 'ci_lower': ci_train[0],
                         'ci_upper': ci_train[1]}
        new_row_test = {'hidden_layers': layer_num, 'mean': np.mean(hid_layers_acc_test), 'ci_lower': ci_test[0],
                        'ci_upper': ci_test[1]}
        hid_layers_df_train = pd.concat([hid_layers_df_train, pd.DataFrame([new_row_train])], ignore_index=True)
        hid_layers_df_test = pd.concat([hid_layers_df_test, pd.DataFrame([new_row_test])], ignore_index=True)

    # Save to csv
    hid_layers_df_train = hid_layers_df_train.astype({"hidden_layers": float})
    hid_layers_df_test = hid_layers_df_test.astype({"hidden_layers": float})
    print(hid_layers_df_test)
    hid_layers_df_train.to_csv('hid_layers_df_train.csv', index=False)
    hid_layers_df_test.to_csv('hid_layers_df_test.csv', index=False)

    # Plot accuracy vs no. of hidden layers
    plot_acc(hid_layers_df_train, hid_layers_df_test, 'Accuracy VS No. of Hidden Layers', 'hidden_layers')

    # 5.) Find optimum L2 regularization term
    l2_df_train = pd.DataFrame(columns=['alpha', 'mean', 'ci_lower', 'ci_upper'])
    l2_df_test = pd.DataFrame(columns=['alpha', 'mean', 'ci_lower', 'ci_upper'])
    for alpha in (0.0001, 0.001, 0.01, 0.1, 1, 10):
        l2_acc_train = []
        l2_acc_test = []
        for i in range(exp):
            acc_train, acc_test = nn_classifier(
                data_x, data_y,
                percent_test=0.4,
                exp_num=i,
                epoch=1000,
                normalise=True,
                hidden=(90, 90, 90),
                solver='sgd',
                L2=alpha,
                learn_rate=0.05,
                plot_roc=False)
            l2_acc_train.append(acc_train)
            l2_acc_test.append(acc_test)

        # Calculate confidence interval for test and train
        ci_train = confidence_interval(l2_acc_train, 0.95)
        ci_test = confidence_interval(l2_acc_test, 0.95)

        # Add data to dataframe for plotting later
        new_row_train = {'alpha': alpha, 'mean': np.mean(l2_acc_train), 'ci_lower': ci_train[0],
                         'ci_upper': ci_train[1]}
        new_row_test = {'alpha': alpha, 'mean': np.mean(l2_acc_test), 'ci_lower': ci_test[0],
                        'ci_upper': ci_test[1]}
        l2_df_train = pd.concat([l2_df_train, pd.DataFrame([new_row_train])], ignore_index=True)
        l2_df_test = pd.concat([l2_df_test, pd.DataFrame([new_row_test])], ignore_index=True)

    # Save to csv
    l2_df_train = l2_df_train.astype({"alpha": float})
    l2_df_test = l2_df_test.astype({"alpha": float})
    print(l2_df_test)
    l2_df_train.to_csv('l2_df_train.csv', index=False)
    l2_df_test.to_csv('l2_df_test.csv', index=False)

    # Plot accuracy vs no. of hidden layers
    plot_acc(l2_df_train, l2_df_test, 'Accuracy VS L2 Strength', 'alpha')

    # 6.) Adam vs SGD
    opt_df_train = pd.DataFrame(columns=['optimizer', 'mean', 'ci_lower', 'ci_upper'])
    opt_df_test = pd.DataFrame(columns=['optimizer', 'mean', 'ci_lower', 'ci_upper'])
    for optimizer in ('sgd', 'adam'):
        opt_acc_train = []
        opt_acc_test = []
        for i in range(exp):
            acc_train, acc_test = nn_classifier(
                data_x, data_y,
                percent_test=0.4,
                exp_num=i,
                epoch=1000,
                normalise=True,
                hidden=(90, 90, 90),
                solver=optimizer,
                L2=0,
                learn_rate=0.05,
                plot_roc=False)
            opt_acc_train.append(acc_train)
            opt_acc_test.append(acc_test)

        # Calculate confidence interval for test and train
        ci_train = confidence_interval(opt_acc_train, 0.95)
        ci_test = confidence_interval(opt_acc_test, 0.95)

        # Add data to dataframe for plotting later
        new_row_train = {'optimizer': optimizer, 'mean': np.mean(opt_acc_train), 'ci_lower': ci_train[0],
                         'ci_upper': ci_train[1]}
        new_row_test = {'optimizer': optimizer, 'mean': np.mean(opt_acc_test), 'ci_lower': ci_test[0],
                        'ci_upper': ci_test[1]}
        opt_df_train = pd.concat([opt_df_train, pd.DataFrame([new_row_train])], ignore_index=True)
        opt_df_test = pd.concat([opt_df_test, pd.DataFrame([new_row_test])], ignore_index=True)

    # Save to csv
    opt_df_train = opt_df_train.astype({"optimizer": str})
    opt_df_test = opt_df_test.astype({"optimizer": str})
    opt_df_train['error'] = opt_df_train['ci_upper'] - opt_df_train['ci_lower']
    opt_df_test['error'] = opt_df_test['ci_upper'] - opt_df_test['ci_lower']
    print(opt_df_test)
    opt_df_train.to_csv('opt_df_train.csv', index=False)
    opt_df_test.to_csv('opt_df_test.csv', index=False)

    # Plot accuracy vs no. of hidden layers
    plot_bar(opt_df_train, opt_df_test, 'Accuracy - SGD vs Adam', 'optimizer')

    # 7.) Best Model
    best_df_train = pd.DataFrame(columns=['best_model', 'mean', 'ci_lower', 'ci_upper'])
    best_df_test = pd.DataFrame(columns=['best_model', 'mean', 'ci_lower', 'ci_upper'])
    best_acc_train = []
    best_acc_test = []
    best_cm = []
    for i in range(exp):
        acc_train, acc_test, cm = nn_classifier(
            data_x, data_y,
            percent_test=0.4,
            exp_num=i,
            epoch=1000,
            normalise=True,
            hidden=(90, 90, 90),
            solver='sgd',
            L2=0,
            learn_rate=0.05,
            plot_roc=True)
        best_acc_train.append(acc_train)
        best_acc_test.append(acc_test)
        best_cm.append(cm)

    # Get best confusion matrix
    print('Best confusion matrix is')
    print(best_cm[best_acc_test.index(max(best_acc_test))])

    # Calculate confidence interval for test and train
    ci_train = confidence_interval(best_acc_train, 0.95)
    ci_test = confidence_interval(best_acc_test, 0.95)

    # Add data to dataframe for plotting later
    new_row_train = {'best_model': 'best_model', 'mean': np.mean(best_acc_train), 'ci_lower': ci_train[0],
                     'ci_upper': ci_train[1]}
    new_row_test = {'best_model': 'best_model', 'mean': np.mean(best_acc_test), 'ci_lower': ci_test[0],
                    'ci_upper': ci_test[1]}
    best_df_train = pd.concat([best_df_train, pd.DataFrame([new_row_train])], ignore_index=True)
    best_df_test = pd.concat([best_df_test, pd.DataFrame([new_row_test])], ignore_index=True)

    # Save to csv
    best_df_train = best_df_train.astype({"best_model": str})
    best_df_test = best_df_test.astype({"best_model": str})
    best_df_train['error'] = best_df_train['ci_upper'] - best_df_train['ci_lower']
    best_df_test['error'] = best_df_test['ci_upper'] - best_df_test['ci_lower']
    print('Best Model Accuracy')
    print(best_df_test)
    best_df_train.to_csv('best_df_train.csv', index=False)
    best_df_test.to_csv('best_df_test.csv', index=False)

    # Plot accuracy vs no. of hidden layers
    plot_bar(best_df_train, best_df_test, 'Accuracy - Best Model', 'best_model')


    # Part B Neural Network Regression
    reg_data_x, reg_data_y = get_data('abalone.data')
    rmse_df = pd.DataFrame(columns=['optimizer', 'mean', 'ci_lower', 'ci_upper'])
    r2_df = pd.DataFrame(columns=['optimizer', 'mean', 'ci_lower', 'ci_upper'])
    rmse_train_df = pd.DataFrame(columns=['optimizer', 'mean', 'ci_lower', 'ci_upper'])
    r2_train_df = pd.DataFrame(columns=['optimizer', 'mean', 'ci_lower', 'ci_upper'])

    for optimizer in ('sgd', 'adam'):
        print(f'Neural Network Regression: {optimizer}')
        rmse_list = []
        r2_list = []
        rmse_train_list = []
        r2_train_list = []
        for i in range(exp):
            rmse, r2, rmse_train, r2_train = nn_reg(
                reg_data_x, reg_data_y,
                percent_test=0.4,
                exp_num=i,
                normalise=True,
                hidden=(100,),
                solver=optimizer,
                epoch=500,
                learn_rate=0.005)
            rmse_list.append(rmse)
            r2_list.append(r2)
            rmse_train_list.append(rmse_train)
            r2_train_list.append(r2_train)

        # Calculate confidence interval for test and train
        rmse_ci = confidence_interval(rmse_list, 0.95)
        r2_ci = confidence_interval(r2_list, 0.95)
        rmse_train_ci = confidence_interval(rmse_train_list, 0.95)
        r2_train_ci = confidence_interval(r2_train_list, 0.95)
        new_row_rmse = {'optimizer': optimizer, 'mean': np.mean(rmse_list), 'ci_lower': rmse_ci[0],
                         'ci_upper': rmse_ci[1]}
        new_row_r2 = {'optimizer': optimizer, 'mean': np.mean(r2_list), 'ci_lower': r2_ci[0],
                        'ci_upper': r2_ci[1]}
        new_row_rmse_train = {'optimizer': optimizer, 'mean': np.mean(rmse_train_list), 'ci_lower': rmse_train_ci[0],
                         'ci_upper': rmse_train_ci[1]}
        new_row_r2_train = {'optimizer': optimizer, 'mean': np.mean(r2_train_list), 'ci_lower': r2_train_ci[0],
                        'ci_upper': r2_train_ci[1]}
        rmse_df = pd.concat([rmse_df, pd.DataFrame([new_row_rmse])], ignore_index=True)
        r2_df = pd.concat([r2_df, pd.DataFrame([new_row_r2])], ignore_index=True)
        rmse_train_df = pd.concat([rmse_train_df, pd.DataFrame([new_row_rmse_train])], ignore_index=True)
        r2_train_df = pd.concat([r2_train_df, pd.DataFrame([new_row_r2_train])], ignore_index=True)

    # Save to csv
    rmse_df['error'] = rmse_df['ci_upper'] - rmse_df['ci_lower']
    r2_df['error'] = r2_df['ci_upper'] - r2_df['ci_lower']
    rmse_train_df['error'] = rmse_train_df['ci_upper'] - rmse_train_df['ci_lower']
    r2_train_df['error'] = r2_train_df['ci_upper'] - r2_train_df['ci_lower']
    print(rmse_df)
    print(r2_df)
    rmse_df.to_csv('rmse_df.csv', index=False)
    r2_df.to_csv('r2_df.csv', index=False)
    rmse_train_df.to_csv('rmse_train_df.csv', index=False)
    r2_train_df.to_csv('r2_train_df.csv', index=False)


    # Plot accuracy vs no. of hidden layers
    plot_bar(rmse_train_df, rmse_df, 'RMSE - SGD vs Adam', 'optimizer')
    plot_bar(r2_train_df, r2_df, 'R2 - SGD vs Adam', 'optimizer')


if __name__ == '__main__':
     main()
