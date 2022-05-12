from re import L
from data_preprocessing import preprocess
from gradient_descent import LogisticRegression

def driver(mode):
    x_train, x_test, y_train, y_test = preprocess()
    if mode == "gd":
        gradient_descent = LogisticRegression(learning_rate = 0.1)
        gradient_descent.fit(x=x_train, y=y_train, epochs = 100, verbose = True, early_stop=True)
        test_acc = gradient_descent.test(x_test, y_test)
        print(f'testing accuracy is: {test_acc}')
    elif mode == "newton":
        pass
    else:
        raise Exception(f'Incorrect mode name for driver. Got: {mode}')

if __name__ == "__main__":
    mode = "gd"
    driver(mode)
