import email
from re import L
from data_preprocessing import preprocess
from gradient_descent import LogisticRegression
from newton_method import NewtonMethod
import matplotlib.pyplot as plt

dataset_path = "dataset"
def driver(mode, epoch_num, dataset_name):
    x_train, x_test, y_train, y_test = preprocess(dataset_path, dataset_name)
    print(f'{x_train.shape}\n')
    if mode == "gd":
        gradient_descent = LogisticRegression(learning_rate = 0.1)
        # x_train, y_train = x_train.to_numpy(), y_train.to_numpy()
        _, [train_accuracies,losses] = gradient_descent.fit(x=x_train, y=y_train, epochs = epoch_num, verbose = True, early_stop=True)
        test_acc = gradient_descent.test(x_test, y_test)
        print(f'testing accuracy is: {test_acc}')
        plt.figure()
        plt.title(f'Training Accuracy for {dataset_name} Using Gradient Descent')
        plt.plot(range(len(train_accuracies)), train_accuracies)
        plt.xlabel("number of epoch")
        plt.ylabel("Training Accuracys")
        plt.savefig(f"{mode}_{dataset_name}.png")

    elif mode == "newton":
        newton = NewtonMethod()
        y_train =y_train.to_numpy()
        _, [train_accuracies,losses] = newton.fit(x=x_train, y=y_train, epoch = epoch_num, verbose = True)
        plt.figure()
        plt.title(f'Training Accuracy for {dataset_name} Using Gradient Descent')
        plt.plot(range(len(train_accuracies)), train_accuracies)
        plt.xlabel("number of epoch")
        plt.ylabel("Training Accuracys")
        plt.savefig(f"{mode}_{dataset_name}.png")
    else:
        raise Exception(f'Incorrect mode name for driver. Got: {mode}')

if __name__ == "__main__":
    mode = "gd"
    epoch_num = 100
    dataset_name = "email"
    driver(mode, epoch_num, dataset_name)
