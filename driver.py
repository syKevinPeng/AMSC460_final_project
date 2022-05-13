import email
from re import L
from data_preprocessing import preprocess
from gradient_descent import LogisticRegression
from newton_method import NewtonMethod

dataset_path = "dataset"
def driver(mode, epoch_num, dataset_name):
    x_train, x_test, y_train, y_test = preprocess(dataset_path, dataset_name)
    print(f'{x_train.shape}\n')
    if mode == "gd":
        gradient_descent = LogisticRegression(learning_rate = 0.1)
        # x_train, y_train = x_train.to_numpy(), y_train.to_numpy()
        gradient_descent.fit(x=x_train, y=y_train, epochs = epoch_num, verbose = True, early_stop=True)
        test_acc = gradient_descent.test(x_test, y_test)
        print(f'testing accuracy is: {test_acc}')
    elif mode == "newton":
        newton = NewtonMethod()
        y_train =y_train.to_numpy()
        param = newton.fit(x=x_train, y=y_train, epoch = epoch_num, verbose = True)
        print(f'parameter: {param}')
    else:
        raise Exception(f'Incorrect mode name for driver. Got: {mode}')

if __name__ == "__main__":
    mode = "newton"
    epoch_num = 100
    dataset_name = "bank"
    driver(mode, epoch_num, dataset_name)
