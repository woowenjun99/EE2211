import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder

########################################
# DO NOT modify the following fucntions 
########################################
def make_dataset():
    """
    Generate a 3-class dataset for classification
    """
    def make_hole(x, x1_0, x2_0, r):
        assert (len(x.shape)==2) and (x.shape[1] == 2)
        mask = (x[:,0] - x1_0)**2 + (x[:,1] - x2_0)**2 >= r**2
        return x[mask]
    N_per_cluster = 1000
    short, _long, r_hole = 0.3, 0.4, 0.3
    class1a = np.random.normal(loc=(0,1), scale=(short, _long), size=(N_per_cluster, 2)) # scale=(x, y) = (width,height)
    class1b = np.random.normal(loc=(0,-1), scale=(short, _long), size=(N_per_cluster, 2))
    class1a = make_hole(class1a, 0, 1, r_hole)
    class1b = make_hole(class1b, 0, -1, r_hole)

    class2a = np.random.normal(loc=(-1,0), scale=(_long, short), size=(N_per_cluster, 2))
    class2b = np.random.normal(loc=(0, 1), scale=(r_hole/5, r_hole/5), size=(int(N_per_cluster), 2))

    class3a = np.random.normal(loc=(1, 0), scale=(_long, short), size=(N_per_cluster, 2))
    class3b = np.random.normal(loc=(0,-1), scale=(r_hole/5, r_hole/5), size=(int(N_per_cluster), 2))

    X = np.vstack([class1a, class1b, class2a, class2b, class3a, class3b])
    Y = np.hstack([np.zeros(len(class1a)), np.zeros(len(class1b)), np.ones(len(class2a)), np.ones(len(class2b)),  np.ones(len(class3a))*2,  np.ones(len(class3b))*2]).astype(np.uint)

    return X, Y

def plot_decision_boundary(pred_function, X):
    x_min, x_max = X[:,0].min() - 0.1, X[:,0].max()+0.1
    y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
    h = 0.01, 
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cmap = colors.ListedColormap(['#aa000022','#0000aa22','#00aa0022'])
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=cmap)
########################################
# DO NOT modify the above fucntions 
########################################

def A2_A0239038B(enable_visualization=True): # DO NOT modify the parameter enable_visualization as it will be used during grading.
    # print("Start processing now")
    X, Y = make_dataset()
    X_train = None 
    Y_train = None
    X_test = None
    Y_test = None
    #######################################################################################################
    # Task 1: Data Splitting
    # TODO: split the dataset into two sets: 70% of samples for training, and 30% of samples for testing.
    ########################################################################################################
    
    train_size = 0.7
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=train_size, random_state=42)

    #######################################################################################################
    # End of your code for task 1
    #######################################################################################################

    # visualize the generated dataset (Optional)
    if enable_visualization:
        try:
            color_dict = {0: 'red', 1: 'green', 2: 'blue'}
            plt.figure()
            for c in range(3):
                plt.title("train set")
                indices = (Y_train==c)
                plt.scatter(X_train[indices, 0], X_train[indices, 1], s=2, c=color_dict[c])
            plt.savefig('train_set.png')
            plt.figure()
            for c in range(3):
                plt.title("test set")
                indices = (Y_test==c)
                plt.scatter(X_test[indices, 0], X_test[indices, 1], s=2, c=color_dict[c])
            plt.savefig('test_set.png')
        except: plt.close()

    Y_train_onehot = None
    Y_test_onehot = None
    #######################################################################################################
    # Task 2: One-hot Encoding
    # TODO: Generate one-hot encoded labels from Y_train and Y_test
    #######################################################################################################

    encoder = OneHotEncoder(sparse=False)
    Y_train_onehot = encoder.fit_transform([[x + 1] for x in Y_train])
    Y_test_onehot = encoder.fit_transform([[x + 1] for x in Y_test])
    
    #######################################################################################################
    # End of your code for task 2
    #######################################################################################################

    # Please find the best polynomial order in the loop and fill the following varaibles 
    best_order = None
    best_P_train = None
    best_P_test = None
    best_wp = None
    best_train_acc = 0
    best_test_acc = 0

    for order in range(1, 31):
        #######################################################################################################
        # Task 3: Polynomial Features
        # TODO: Generate polynomial features P_train and P_test from X_train and X_test
        ########################################################################################################
        # print(f"Processing Order {order}")
        poly = PolynomialFeatures(order)
        P_train = np.array(poly.fit_transform(X_train))
        P_test = np.array(poly.fit_transform(X_test))
        
        #######################################################################################################
        # End of your code for task 3
        #######################################################################################################
        

        #######################################################################################################
        # Task 4: Classification
        # TODO: estimate the coefficients wp on the training set P_train and then measure the 
        #       classification accuracy for both training and testing sets. At the end of the loop, you 
        #       need to find the best classifier according to the accuracy.
        ########################################################################################################

        alpha = 0.0001
        num_rows, num_cols = P_train.shape
        wp = None
        if num_rows <= num_cols:
            wp = P_train.T @ np.linalg.inv(P_train @ P_train.T + alpha * (np.identity(num_rows))) @ Y_train_onehot
        else:
            wp = np.linalg.inv(P_train.T @ P_train + alpha * (np.identity(num_cols))) @ P_train.T @ Y_train_onehot

        output_from_my_classifier = P_train @ wp
        pred_labels = np.argmax(output_from_my_classifier, axis=1)
        train_acc = np.mean(pred_labels == Y_train)
        test_acc = np.mean(np.argmax(P_test @ wp, axis=1) == Y_test)

        if test_acc > best_test_acc: # Find the best classifier
           best_order = order
           best_P_train = P_train
           best_P_test = P_test
           best_wp = wp 
           best_train_acc = train_acc
           best_test_acc = test_acc

        #######################################################################################################
        # End of your code for task 4
        #######################################################################################################

        # Visualize the decision boundary (DO NOT modify)
        if enable_visualization:
            try:
                color_dict = {0: 'red', 1: 'blue', 2: 'green'}
                plt.figure()
                plot_decision_boundary(lambda x: poly.fit_transform(x).dot(wp).argmax(1), X=X_train)
                for c in range(3):
                    indices = (Y_train==c)
                    plt.scatter(X_train[indices, 0], X_train[indices, 1], s=2, c=color_dict[c])
                plt.title("order=%d"%order)
                plt.savefig('decision_boundary_order=%d.png'%(order))
                plt.close()
            except: plt.close()
                

    # print(f"The highest test accuracy is {best_test_acc} and the order is {best_order}")
    return X_train, Y_train, X_test, Y_test, Y_train_onehot, Y_test_onehot, best_order, best_P_train, best_P_test, best_wp, best_train_acc, best_test_acc

# main calling code
if __name__=='__main__':
    X_train, Y_train, X_test, Y_test, Y_train_onehot, Y_test_onehot, best_order, best_P_train, best_P_test, best_wp, best_train_acc, best_test_acc = A2_A0239038B(enable_visualization=True) # enable visualization for debug            