from mnist import load_mnist
import numpy as np
# import pickle


# def get_data():
#     (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
#     return x_test, t_test


# def init_network():
#     with open("sample_weight.pkl", "rb") as f:
#         network = pickle.load(f)
#     return network


# x, t = get_data()
# network = init_network()

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
# print(x_train.shape)
# print(t_train.shape)
train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)

# print(batch_mask)

x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]
