from tensorflow import random
from keras.models import Sequential
from keras.layers import GRU, Dense
from numpy import float32, abs, mean, std


def gen_z(batch_size, z_len, z_dim):
    yield random.uniform((batch_size, z_len, z_dim))


def make_net(n_layers, hidden_units, output_units, net_name):
    net = Sequential(name=net_name)
    for layer in range(n_layers):
        net.add(GRU(hidden_units, return_sequences=True, name=f'gru_{layer}'))
    net.add(Dense(output_units, 'sigmoid', name='dense_0'))
    return net


def course_fine_vol(data):
    c_vol, f_vol = float32(), float32()
    for day in range(1, 6):
        c_vol += data.shift(day)
        f_vol += abs(data.shift(day))
    return abs(c_vol).dropna(), f_vol.dropna()


def course_fine_corr(data, lag):
    c_vol, f_vol = course_fine_vol(data)
    num = mean((c_vol.shift(-lag) - mean(c_vol)) * (f_vol - mean(f_vol)))
    den = std(c_vol) * std(f_vol)
    return num / den
