from keras.utils import set_random_seed
from tensorflow import config, convert_to_tensor
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from pandas import read_csv, DataFrame
from p1_0_utils import *
from tensorflow import tile, expand_dims, reshape, random, concat
from numpy.random import choice
from numpy import min, max
from matplotlib.pyplot import figure, plot, xlabel, ylabel, legend, gcf, close


def p2_0_sims():
    folder = './simulations/'
    sequence_len = 12
    condition_len = 7
    window_len = 257
    n_sims = 10000

    set_random_seed(42)
    config.experimental.enable_op_determinism()
    Path(folder).mkdir(parents=True, exist_ok=True)
    mms = MinMaxScaler()

    stock_px = read_csv('./raw/stock_px.csv', index_col=[0], parse_dates=[0])
    stock_len, f_dim = stock_px.shape

    units = f_dim * 4
    gen_net = make_net(2, units, units, 'generator')
    sup_net = make_net(2, units, units, 'supervisor')
    rec_net = make_net(3, units, f_dim, 'recovery')

    gen_len = sequence_len - condition_len
    gen_net.build(input_shape=(None, gen_len, f_dim + condition_len * f_dim))
    sup_net.build(input_shape=(None, gen_len, units))
    rec_net.build(input_shape=(None, gen_len, units))

    sim_px, sim_pf = list(), list()
    window = 0
    while window < stock_len - window_len:
        gen_net.load_weights(f'./models/w{window}_g.h5')
        sup_net.load_weights(f'./models/w{window}_s.h5')
        rec_net.load_weights(f'./models/w{window}_r.h5')

        window_px = stock_px[window:window + window_len]
        final_px = window_px.iloc[-1].values

        window_diff = window_px.diff().dropna()
        scaled_diff = mms.fit_transform(window_diff)
        scaled_data = convert_to_tensor(scaled_diff, float32)

        c = scaled_data[-sequence_len:][1:1 + condition_len]
        c = tile(expand_dims(c, 0), (n_sims, 1, 1))
        c = reshape(c, (n_sims, -1))
        c = reshape(tile(c, (1, gen_len)), (n_sims, gen_len, -1))
        z = random.uniform((n_sims, gen_len, f_dim))
        cz = concat((c, z), -1)

        h_hat = gen_net(cz)
        h_hat_s = sup_net(h_hat)
        x_hat = rec_net(h_hat_s)

        sim_px_list = list()
        for sim in range(n_sims):
            sim_diff = mms.inverse_transform([x_hat[sim][-1]])
            sim_px_list.append(final_px + sim_diff[0])
        sim_px.append(sim_px_list[choice(n_sims)])
        sim_pf.append(mean(sim_px_list, 1))

        print(f'Simulation Window: {window:4d}')
        window += 1

    sim_px = DataFrame(sim_px, stock_px[window_len:].index, stock_px.columns)
    sim_pf = DataFrame(sim_pf, stock_px[window_len:].index)

    figure(figsize=(10, 5))
    plot(max(sim_pf, 1), c='blue', label='Simulated Maximum Price')
    plot(mean(stock_px[window_len:], 1), c='black', label='Realized Price')
    plot(min(sim_pf, 1), c='red', label='Simulated Minimum Price')
    xlabel('Date')
    ylabel('Price')
    legend()
    plt_pf = gcf()
    close()

    round(sim_px, 5).to_csv(f'{folder}sim_px.csv')
    round(sim_pf, 5).to_csv(f'{folder}sim_pf.csv')
    plt_pf.savefig(f'{folder}plt_sim_pf.pdf', bbox_inches='tight')


if __name__ == '__main__':
    p2_0_sims()
