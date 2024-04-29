import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn import neighbors, datasets, model_selection
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
# Main work here:
def mlp_digit(
    hidden,
    iterations,
    X_train,
    y_train,
    X_val,
    y_val,
):
    
    # --------------------
    # Train the MLP
    # --------------------

    # Create an instance of the MLP class for current value of hidden & iterations:

    MLP = MLPClassifier(hidden_layer_sizes=(hidden,), max_iter=iterations, random_state=0)
    # Train the classifier with the training data

    MLP.fit(X_train, y_train)
    
    # --------------------
    # Model accuracy:
    # --------------------

    # Accuracy on train set:
    train_predictions = MLP.predict(X_train)
    good_train_predictions = (train_predictions == y_train)
    train_accuracy = np.sum(good_train_predictions) / len(y_train)
    # Accuracy on validation set:
    val_predictions = MLP.predict(X_val)
    good_val_predictions = (val_predictions == y_val)
    val_accuracy = np.sum(good_val_predictions) / len(y_val)
    return (MLP, train_accuracy, val_accuracy)

# Main work here:
def dt_digit(
    criterion,
    splitter,
    depth,
    X_train,
    y_train,
    X_val,
    y_val,
):
    
    # --------------------
    # Train the MLP
    # --------------------

    # Create an instance of the DecisionTree class for current values:

    dt = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=depth) 
    # Train the classifier with the training data

    dt.fit(X_train, y_train)
    
    # --------------------
    # Model accuracy:
    # --------------------

    # Accuracy on train set:
    train_predictions = dt.predict(X_train)
    good_train_predictions = (train_predictions == y_train)
    train_accuracy = np.sum(good_train_predictions) / len(y_train)
    # Accuracy on validation set:
    val_predictions = dt.predict(X_val)
    good_val_predictions = (val_predictions == y_val)
    val_accuracy = np.sum(good_val_predictions) / len(y_val)
    return (dt, train_accuracy, val_accuracy)



#We will create a class for the SVM classifier, using the one provided by sklearn

class SVM_classifier:
    def __init__(self, kernel='linear', random_state=0,degree=3):
        #Random statate is used to make sure that the results are reproducible
        self.kernel = kernel
        self.random_state = random_state
        self.degree = degree
        self.classifier = SVC(kernel=self.kernel, random_state=self.random_state, degree=degree)
        

    def fit(self, X, y):
        self.classifier.fit(X, y)
    
    def predict(self, X,y):
        predictions = self.classifier.predict(X)
        accuracy=accuracy_score(y, predictions)

        return predictions, accuracy
    

    
    #We will create a function to generate different models with different kernels
    def generate_models(self):
        print('Generating models')
        #We will create a dictionary to store the different models
        models = {}
        #We will create a list with the different kernels we want to test
        kernels = ['poly', 'rbf', 'sigmoid']

        #We will iterate over the list of kernels
        for kernel in kernels:
            if kernel=='poly':
                for degree in range(1,self.degree+1):
                    #We will create a model for each degree of the polynomial kernel
                    model = SVM_classifier(kernel=kernel, degree=degree)
                    models[kernel+str(degree)] = model
            else:
                #We will create a model for each other kernel
                model = SVM_classifier(kernel=kernel)
                models[kernel] = model
        
        print('Models generated')
        self.models=models

    def train_models(self, X_train, y_train):
        print('Training models')
        #We will train each model
        for model in self.models:
            print('Training model: '+model)
            self.models[model].fit(X_train, y_train)
        print('Models trained')
     

    def choose_best_model(self, X_validation, y_validation):
        #We will create a dictionary to store the scores of each model
        scores = {}
        #We will iterate over the models
        for model in self.models:
            #We will store the score of each model
            predictions,score= self.models[model].predict(X_validation, y_validation)
            print('Model: '+model+' Score: '+str(score))
            scores[model] = score

        #We create a graph to compare the scores of the different models
        plt.bar(range(len(scores)), list(scores.values()), align='center')
        plt.xticks(range(len(scores)), list(scores.keys()))
        plt.xlabel('Model')
        plt.ylabel('Accurracy')
        plt.title('Scores of the different models')

        plt.show(block=False)
        plt.pause(10)
        plt.close()
        #We will choose the model with the highest score
        best_model = max(scores, key=scores.get)
        return best_model, scores[best_model]




#We load the data
X = np.load('emnist_hex_images.npy')
y = np.load('emnist_hex_labels.npy')

seed = 0



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

#We split the test data into test and validation
X_validation, X_test, y_validation, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=seed)


#We create a SVM object
svm=SVM_classifier(degree=3)
#We generate the models
svm.generate_models()

#We train the models
svm.train_models(X_train, y_train)

#We choose the best model
best_model, score = svm.choose_best_model(X_validation, y_validation)

#We store the best model in a dictionary with its score
best_models = {}
best_models['SVM'] = (svm.models[best_model], score)



# Create vectors to store the results for each k:
MLP_train_accuracies = []
MLP_val_accuracies = []

# Train a MLP for each value of hidden layers and iterations
hidden_list = [1, 3, 5, 7, 9]
iterations_list = [1, 10, 100, 1000, 10000]

MLP_models = []
params_list = [(h, i, X_train, y_train, X_validation, y_validation) for h in hidden_list for i in iterations_list]

with ThreadPoolExecutor(max_workers=6) as executor:
    results_mlp = list(executor.map(lambda p: mlp_digit(*p), params_list))

for MLP,train_acc,val_acc in results_mlp:
        print(f"MLP classifier trained with {MLP.get_params()['hidden_layer_sizes']} hidden layers & {MLP.get_params()['max_iter']} maximum iterations")
        print(f"Train accuracy: {train_acc:.5f}")
        print(f"Validation accuracy: {val_acc:.5f}")
        MLP_train_accuracies.append(train_acc)
        MLP_val_accuracies.append(val_acc)
        MLP_models.append(MLP)

# Create vectors to store the results for each k:
DecisionTree_train_accuracies = []
DecisionTree_val_accuracies = []

# Train a MLP for each value of hidden layers and iterations
criterion = ['gini', 'entropy', 'log_loss']
splitter = ['best','random']
max_depth = [16, 20, 30, 40, 999999999]


dt_models = []
params_list_dt = [(c, s, d, X_train, y_train, X_validation, y_validation) for c in criterion for s in splitter for d in max_depth]

with ThreadPoolExecutor(max_workers=6) as executor:
    results_dt = list(executor.map(lambda p: dt_digit(*p), params_list_dt))
    

for dt,train_acc,val_acc in results_dt:
                    print(f"DT classifier trained with {dt.get_params()['criterion']} criterion & {dt.get_params()['splitter']} splitter & {dt.get_params()['max_depth']} max_depth")
                    print(f"Train accuracy: {train_acc:.5f}")
                    print(f"Validation accuracy: {val_acc:.5f}")
                    DecisionTree_train_accuracies.append(train_acc)
                    DecisionTree_val_accuracies.append(val_acc)
                    dt_models.append(dt)


# Automatically select the best model based on validation accuracies

mlp_selected = np.argmax(MLP_val_accuracies)
dt_selected = np.argmax(DecisionTree_val_accuracies)

best_models['MLP']=(mlp_selected, MLP_val_accuracies[mlp_selected])
best_models['DT']=(dt_selected, DecisionTree_val_accuracies[dt_selected])

#Now, we find the best model among the SVM, MLP and Decision Tree
best_key,best_model = max(best_models.items(), key=lambda x: x[1][1])
print('The best model is: '+best_key)
print('The score of the best model is: '+str(best_model[1]))

#We test the best model
print('Testing best model')
if(best_key=='SVM'):
    predictions, score = best_model[0].predict(X_test, y_test)
    print('Best model score: '+str(score))
    
else:
    predictions = best_model[0].predict(X_test)
    score = accuracy_score(y_test, predictions)
    print('Best model score: '+str(score))

print('The confusion matrix is:')
cm=confusion_matrix(y_test, predictions)    
dp=ConfusionMatrixDisplay(cm,display_labels=['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','empty'])
dp.plot()
plt.show(block=False)
plt.pause(10)
plt.close()








