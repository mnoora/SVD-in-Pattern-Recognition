import svd
import matplotlib.pyplot as plt
import numpy as np

def main():
    x_train, t_train,x_test,t_test = svd.get_mnist()

    #Applying SVD to numbers between 0 and 9
    U_all_values = svd.singular_value_decomposition_for_all_numbers(x_train,t_train)

    # Determine the optimal value of the cut-off parameter k
    #k = svd.determine_cut_off_k(U_two_values[0],U_two_values[1],x_test,t_test)
    k = 7

    #Checking how many digits (from 0 to 9) are classified right
    for i in range(10):
        svd.calculate_how_successfull_classification_is(500,i,x_test,t_test,U_all_values,k)


    # Applying SVD to 4 and 6, and visualizing first 4 eigenpatterns
    U_two_values = svd.singular_value_decomposition_for_two_numbers(x_train,t_train,4,6)
    svd.visualize_eigenpatterns(U_two_values[0]['u'],U_two_values[1]['u'])


    # Checking how successful classification was for digit 4
    svd.calculate_how_successfull_classification_is(50,4,x_test,t_test,U_two_values,20)

    # Checking how successful classification was for digit 6
    svd.calculate_how_successfull_classification_is(50,6,x_test,t_test,U_two_values,20)

    test_images = svd.download_images()
    svd.classify_three_test_images_not_from_MNIST(test_images,x_train,t_train,20)


if __name__=="__main__":
    main()