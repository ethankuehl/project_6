import sklearn.metrics as metrics

def multiclass_classification_metrics(gs, X_test, y_test):
    
    y_hat = gs.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_hat)

    #specificity = true negative/(true neagtive +false positive)
    specificity = 1984/(1984+21) 

    sensitivity =  metrics.recall_score(y_test, y_hat, average='macro')

    precision = metrics.precision_score(y_test, y_hat, average='macro')

    f1 = metrics.f1_score(y_test, y_hat, average='macro')
    
    
    print('My accuracy is: ', round(accuracy,4))
    print('My specificity is: ', round(specificity, 4))
    print('My sensitivity is: ', round(sensitivity,4))
    print('My precision is: ', round(precision,4))
    print('My f1 score is: ', round(precision,4))
    

    metrics.plot_confusion_matrix(gs, X_test, y_test, cmap='Accent', 
                          values_format='d', display_labels=[ 'Change-up',
                                                              'Breaking Ball', 
                                                             'Fastball'])
    
    plot_multiclass_roc(gs, X_test, y_test, 3, figsize=(17, 6))

def multiclass_classification_metrics(gs, X_test, y_test):
    
    y_hat = gs.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_hat)

    #specificity = true negative/(true neagtive +false positive)
    specificity = 1984/(1984+21) 

    sensitivity =  metrics.recall_score(y_test, y_hat, average='macro')

    precision = metrics.precision_score(y_test, y_hat, average='macro')

    f1 = metrics.f1_score(y_test, y_hat, average='macro')
    
    
    print('My accuracy is: ', round(accuracy,4))
    print('My specificity is: ', round(specificity, 4))
    print('My sensitivity is: ', round(sensitivity,4))
    print('My precision is: ', round(precision,4))
    print('My f1 score is: ', round(precision,4))
    

    metrics.plot_confusion_matrix(gs, X_test, y_test, cmap='Accent', 
                          values_format='d', display_labels=[ 'Change-up',
                                                              'Breaking Ball', 
                                                             'Fastball']);
    
    plot_multiclass_roc(gs, X_test, y_test, 3, figsize=(17, 6))

def binary_classification_metrics(gs, X_test, y_test):

y_hat = gs.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_hat)

#specificity = true negative/(true neagtive +false positive)
specificity = 1984/(1984+21) 

sensitivity =  metrics.recall_score(y_test, y_hat)

precision = metrics.precision_score(y_test, y_hat)

f1 = metrics.f1_score(y_test, y_hat)
print('My accuracy is: ', round(accuracy,4))
print('My specificity is: ', round(specificity, 4))
print('My sensitivity is: ', round(sensitivity,4))
print('My precision is: ', round(precision,4))
print('My f1 score is: ', round(precision,4))

cm = np.array([['True Negative', 'False Positive'],
                        ['False Negative', 'True Positive']])

cm = pd.DataFrame(cm,columns = ['Pred Offspeed', 'Pred Fastball'], 
                    index = ['Actual Offspeed','Actual Fastball'])

# Displaying sample confusion matrix
display(cm)

# Displaying actual confusion matrix 
metrics.plot_confusion_matrix(gs, X_test, y_test, cmap='Accent', 
                        values_format='d', display_labels=['Offspeed Pitch', 
                                                            'Fastball Pitch']);

metrics.plot_roc_curve(gs, X_test, y_test)
# add worst case scenario line
plt.plot([0, 1], [0, 1])
plt.title('ROC AUC Curve')

return f'My ROC AUC score is: {metrics.roc_auc_score(y_test, y_hat)}'