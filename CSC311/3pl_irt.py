from utils import *

import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def log_likelihood(data, theta, beta):
    """ Compute the log-likelihood.
    
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    log_likelihood = 0.
    for n in range(len(data['user_id'])):
        i = data['user_id'][n]
        j = data['question_id'][n]
        is_correct = data['is_correct'][n]
        z = theta[i] - beta[j]
        single_log = is_correct*(z - np.log(1+np.exp(z)))
        single_log -= (1-is_correct)*np.log(1+np.exp(z))
        log_likelihood += single_log
    return log_likelihood


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = log_likelihood(data, theta, beta)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def log_likelihood_3pl(data, theta, beta, c, k):
    """ Compute the log-likelihood for the 3pl-irt.
    
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    log_likelihood = 0.
    for n in range(len(data['user_id'])):
        i = data['user_id'][n]
        j = data['question_id'][n]
        is_correct = data['is_correct'][n]
        z = k[j]*(theta[i] - beta[j])
        single_log = is_correct*(np.log(c+np.exp(z)) - np.log(1+np.exp(z)))
        single_log += (1-is_correct)*(np.log(1-c) - np.log(1+np.exp(z)))
        log_likelihood += single_log
    return log_likelihood


def neg_log_likelihood_3pl(data, theta, beta, c, k):
    """ Compute the negative log-likelihood for the 3pl-irt.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = log_likelihood_3pl(data, theta, beta, c, k)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    for n in range(len(data['user_id'])):
        i = data['user_id'][n]
        j = data['question_id'][n]
        is_correct = data['is_correct'][n]
        d_i = is_correct*np.divide(1, 1+np.exp(theta[i]-beta[j]))
        d_i -= (1-is_correct)*np.divide(np.exp(theta[i]-beta[j]),1+np.exp(theta[i]-beta[j]))
        theta[i] += lr*d_i
        d_j = is_correct*np.divide(-1, 1+np.exp(theta[i]-beta[j]))
        d_j += (1-is_correct)*np.divide(np.exp(theta[i]-beta[j]),1+np.exp(theta[i]-beta[j]))
        beta[j] += lr*d_j
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta

def update_theta_beta_3pl(data, lr, theta, beta, c, k):
    """ Update theta and beta using gradient descent for 3pl-irt.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    for n in range(len(data['user_id'])):
        i = data['user_id'][n]
        j = data['question_id'][n]
        is_correct = data['is_correct'][n]
        z = k[j]*(theta[i] - beta[j])
        d_i = np.divide(k[j]*np.exp(z), 1+np.exp(z))
        d_i = k[j] - d_i
        d_i = is_correct*d_i
        d_i -= (1-is_correct)*np.divide(k[j]*np.exp(z),1+np.exp(z))
        theta[i] += lr*d_i
        d_j = np.divide(k[j]*np.exp(z), 1+np.exp(z))
        d_j -= k[j]
        d_j = is_correct*d_j
        d_j += (1-is_correct)*np.divide(k[j]*np.exp(z),1+np.exp(z))
        beta[j] += lr*d_j
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt_3pl(data, val_data, lr, iterations, c):
    """ Train 3pl IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    theta = np.random.normal(size = 543)
    beta = np.random.normal(size=1774)

    val_acc_lst = []
    training_llk = []
    val_llk = []
    
    for i in range(iterations):
        k = []
        for j in range(1774):
            x = (0.5 - beta[j]).sum()
            p_a = sigmoid(x)
            p_ah = sigmoid(x+0.0001)
            slope = np.divide((p_ah-p_a), 0.0001)
            k.append(slope)
        neg_lld = neg_log_likelihood_3pl(data, theta=theta, beta=beta, c=c, k=k)
        neg_val_lld = neg_log_likelihood_3pl(val_data, theta=theta, beta=beta, c=c, k=k)
        training_llk.append((-1)*neg_lld)
        val_llk.append((-1)*neg_val_lld)
        score = evaluate_3pl(data=val_data, theta=theta, beta=beta, c=c, k=k)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta_3pl(data, lr, theta, beta, c, k)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst, training_llk, val_llk


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    theta = np.random.normal(size = 543)
    beta = np.random.normal(size=1774)

    val_acc_lst = []
    training_llk = []
    val_llk = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        neg_val_lld = neg_log_likelihood(val_data, theta=theta, beta=beta)
        training_llk.append((-1)*neg_lld)
        val_llk.append((-1)*neg_val_lld)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst, training_llk, val_llk


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        #print(p_a)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def evaluate_3pl(data, theta, beta, c, k):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = c + (1-c)*sigmoid(k[q]*x)
        #print(p_a)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")
    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    '''theta, beta, val_acc_lst, training_llk, val_llk = irt(train_data, val_data, 0.003, 200)
    iterations = []
    for i in range(1, 201):
        iterations.append(i)
    plt.xlabel("Iterations")
    plt.ylabel("log-likelihoods")
    plt.plot(iterations, training_llk, label="training log-likelihoods")
    plt.plot(iterations, val_llk, label="validation log-likelihoods")
    plt.legend()
    plt.savefig("part_a_Q2c.png")
    plt.clf()'''
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    # train 3pl irt model
    theta, beta, val_acc_lst, training_llk, val_llk = irt_3pl(train_data, val_data, 0.008, 350, 0.05)
    #####################################################################
    # TODO:                                                             #
    # Final validation and test accuracy
    print("Final validation accuracy is: ", evaluate(val_data, theta, beta))
    print("Final test accuracy is: ", evaluate(test_data, theta, beta))
    #####################################################################
    pass
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
