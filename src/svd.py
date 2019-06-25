import matplotlib.pyplot as plt
import numpy as np

import mnist


def get_mnist():
    """
    Load the MNIST data
    """
    mnist.init()
    x_train, t_train, x_test, t_test = mnist.load()
    print("Loaded MNIST data")
    return x_train,t_train,x_test,t_test


def singular_value_decomposition_for_all_numbers(x_train,t_train):
    """
    Perform Singular Value Decomposition for patterns from 0 to 9. Return a list of dictionaries,
    where every dictionary contains the U matrix and corresponding pattern.

    Keyword arguments:
    x_train -- Training data
    t_train -- Training data labels
    """

    U = []
    # Calculating singular value decomposition for values from 0 to 9
    for i in range(10):

        # Choose only patterns corresponding to value of i
        num_mask = (t_train == i)
        num_dat = x_train[num_mask]

        U1, _, _ =  np.linalg.svd(num_dat.T)

        U.append({'u':U1,'n':i})
        print("Calculated SVD for value =",i)

    return U


def singular_value_decomposition_for_two_numbers(x_train,t_train,num1,num2):
    """
    Perform Singular Value Decomposition for patterns num1 and num2. Return a list of two dictionaries,
    where the dictionaries contain the U matrix and corresponding pattern.

    Keyword arguments:
    x_train -- Training data
    t_train -- Training data labels
    num1    -- Pattern
    num2    -- Pattern
    """

    print("Calculating SVD...")
    U = []

    # Use only images corresponding to the two given patterns num1 and num2
    num_mask1 = (t_train == num1)
    num_dat1 = x_train[num_mask1]

    U1, *_ =  np.linalg.svd(num_dat1.T)

    num_mask2 = (t_train == num2)
    num_dat2 = x_train[num_mask2]

    U2, *_ =  np.linalg.svd(num_dat2.T)

    U.append({'u':U1,'n':num1})
    U.append({'u':U2,'n':num2})

    print("SVD calculated")
    return U


def determine_cut_off_k(U4, U6, x_test, t_test):
    """
    Determine cut-off parameter k by performing classification using SVD with multiple values of k and 
    choosing the value of k when the classification is most successful.

    Keyword arguments:
    U4     -- SVD matrix for digit 4
    U6     -- SVD matrix for digit 6
    x_test -- Testing data
    t_test -- Testing data labels
    """

    accuracies = []
    ks = []
    print("Started calculating optimal k")

    # Perform classification using k values from 1 to 50. By doing this found that k=7 is the optimal value i.e.
    # the classification is most successful then. 
    for k in range(1,50,1):
        success_count_for_4 = calculate_how_successfull_classification_is(500,4,x_test,t_test,[U4,U6],k)
        success_count_for_6 = calculate_how_successfull_classification_is(500,6,x_test,t_test,[U4,U6],k)

        # Calculate accuracy
        acc = (float(success_count_for_4)+float(success_count_for_6))/1000.0
        accuracies.append(acc)
        ks.append(k)
    
    # Plot accuracy as a function of k
    plt.figure(8)
    plt.plot(ks, accuracies)
    plt.ylabel("Accuracy of classification")
    plt.xlabel("cut-off parameter k")
    plt.title("Accuracy as a function of k")
    plt.savefig("AccuracyK1.png")

    # Get k that corresponds to the maximum accuracy
    k = ks[np.argmax(accuracies)]
    print("Optimal k is:",k)
    return k


def calculate_residual(z, U,k):
    """
    Calculate residual for some unknown pattern with some SVD matrix.

    Keyword arguments:
    z -- The unknown pattern (image of digit)
    U -- SVD matrix
    k -- cut-off parameter
    """
    U_k = U[:, 0:k]
    r = np.linalg.norm(np.matmul((1-np.matmul(U_k,U_k.T)),z))
    return r


def visualize_eigenpatterns(U4,U6):
    """
    Visualize four first eigenpatterns for 4 and 6.

    Keyword arguments:
    U4 -- SVD matrix for 4 
    U6 -- SVD matrix for 6
    """
    for i in range(4):
        plt.figure(i)
        plt.imshow(U4[:,i].reshape((28,28)),cmap='gray')
        title = "Eigenpattern " + str(i+1) + " for number 4"
        plt.title(title)
        filename = "num4eig"+str(i+1)
        plt.savefig(filename)

        plt.figure(i+4)
        plt.imshow(U6[:,i].reshape((28,28)),cmap='gray')
        title = "Eigenpattern " + str(i+1) + " for number 6"
        plt.title(title)
        filename = "num6eig"+str(i+1)
        plt.savefig(filename)


def classify(U,z,k):
    """
    Classify the unknown pattern using the SVD matrices.

    Keyword arguments:
    U -- List of SVD matrices 
    z -- The unknown pattern (image of digit)
    """
    r = []
    for u in U:

        r.append(calculate_residual(z,u['u'],k))

    # Find the value corresponding to the smallest residual
    res = U[np.argmin(r)]['n']
    return res


def calculate_how_successfull_classification_is(N,num,x_test,t_test,U,k):
    """
    Calculate the accuracy of the classification. Returns how many successful classifications
    has happened.

    Keyword arguments:
    N      -- The amount of patterns we want to classify
    num    -- The correct pattern
    x_test -- Testing data
    t_test -- Testing data labels
    U      -- SVD matrix for some digit 
    k      -- The cut-off parameter
    """
    num_mask = (t_test == num)

    z_num = x_test[num_mask][0:N]
    count = 0

    # Go through testing data and check how big part of it is classified right
    for z in z_num:
        if classify(U,z,k) == num:
            count +=1

    acc = float(count)/float(N)
    print("Classified pictures of number",num,", accuracy",acc * 100, "%")

    return count


def download_images():
    """
    Downloading images that are not from MNIST (obtained from Internet)
    """

    image_2 = plt.imread('run/digit2.jpg').reshape(784)
    image_2_second = plt.imread('run/digit2_2.jpg').reshape(784)
    image_8 = plt.imread('run/digit8.jpg').reshape(784)

    return [image_2,image_2_second,image_8]


def classify_three_test_images_not_from_MNIST(images,x_train,t_train,k):
    """
    Classifying images that are not from MNIST (obtained from Internet)

    Keyword arguments:
    images  -- Images to be classified
    x_train -- Training data
    t_train -- Training data labels
    k       -- The cut-off parameter
    """
    # Appliying SVD to 2 and 8
    U_2_and_8 = singular_value_decomposition_for_two_numbers(x_train,t_train,2,8)

    # Classifying two images of number 2
    res2_first = classify(U_2_and_8,images[0],k)
    res2_second = classify(U_2_and_8,images[1],k)
    print("First image of two classified as",res2_first,"and the second image classified as",res2_second)

    # Classifying one image of number 8
    res8 = classify(U_2_and_8,images[2],k)
    print("Image of eight classified as",res8)