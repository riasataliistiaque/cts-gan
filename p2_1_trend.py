from sklearn.preprocessing import MinMaxScaler
from pandas import read_csv, DataFrame
from p1_0_utils import *
from tensorflow import convert_to_tensor
from tensorflow import tile, expand_dims, reshape, random, concat
from numpy import cumsum, arange
from matplotlib.pyplot import figure, plot, axvspan, xticks, xlabel, ylabel, legend, gcf, close


def p2_1_trend():
    sequence_len = 12
    condition_len = 7
    window_len = 257
    n_sims = 10000
    folder = './simulations/'

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

    oos_px = stock_px[window_len:]
    oos_diff = oos_px.diff().dropna()
    oos_len = len(oos_diff)

    check_list = [True, False, True, False, True, False, True]
    bear_list, bull_list, side_list = list(), list(), list()
    for stock in stock_px.columns:
        for window in range(oos_len):
            cond_data = oos_diff[window:window + condition_len][f'{stock}']
            if all(cond_data < 0):
                # print(window, cond_data)
                bear_list.append(window)
                break

        for window in range(oos_len):
            cond_data = oos_diff[window:window + condition_len][f'{stock}']
            if all(cond_data > 0):
                # print(window, cond_data)
                bull_list.append(window)
                break

        for window in range(oos_len):
            cond_data = oos_diff[window:window + condition_len][f'{stock}']
            cond_list = (cond_data > 0).tolist()
            if sum([i[0] == i[1] for i in zip(cond_list, check_list)]) == condition_len:
                # print(window, cond_data)
                side_list.append(window)
                break

    for move_index in range(3):
        move_name = ['bear', 'bull', 'side'][move_index]
        move_list = [bear_list, bull_list, side_list][move_index]

        for stock in range(f_dim):
            stock_name = stock_px.columns[stock]
            window = move_list[stock] + sequence_len

            gen_net.load_weights(f'./models/w{window}_g.h5')
            sup_net.load_weights(f'./models/w{window}_s.h5')
            rec_net.load_weights(f'./models/w{window}_r.h5')

            window_px = stock_px[window:window + window_len]
            final_px = window_px.iloc[-sequence_len].values[stock]
            index_px = stock_px[window + window_len - sequence_len + 1:window + window_len + 1].index

            window_diff = window_px.diff().dropna()
            scaled_diff = mms.fit_transform(window_diff)
            scaled_data = convert_to_tensor(scaled_diff, float32)
            cond_data = scaled_data[-sequence_len:][1:1 + condition_len]
            cond_data = tile(expand_dims(cond_data, 0), (n_sims, 1, 1))

            c = reshape(cond_data, (n_sims, -1))
            c = reshape(tile(c, (1, gen_len)), (n_sims, gen_len, -1))
            z = random.uniform((n_sims, gen_len, f_dim))
            cz = concat((c, z), -1)

            h_hat = gen_net(cz)
            h_hat_s = sup_net(h_hat)
            x_hat = rec_net(h_hat_s)
            cx_hat = concat((cond_data, x_hat), 1)

            sim_px_list = list()
            for sim in range(n_sims):
                sim_diff = mms.inverse_transform(cx_hat[sim])
                sim_px_list.append(final_px + cumsum(sim_diff[:, stock]))
            sim_px = DataFrame(sim_px_list).T

            figure(figsize=(20, 10))
            plot(sim_px, lw=0.005, c='blue')
            plt_fake, = plot(stock_px.loc[index_px, stock_name].values, c='blue')  # only for legend purposes
            plt_real, = plot(stock_px.loc[index_px, stock_name].values, c='black')
            plt_cond = axvspan(0, condition_len - 1, color='grey', alpha=0.2)
            plt_is4 = axvspan(condition_len - 1, sequence_len - 2, color='yellow', alpha=0.2)
            plt_oos1 = axvspan(sequence_len - 2, sequence_len - 1, color='orange', alpha=0.2)
            xticks(arange(sequence_len), index_px.strftime('%Y-%m-%d'))
            xlabel('Date')
            ylabel('Price')
            legend(handles=[plt_fake, plt_real, plt_cond, plt_is4, plt_oos1],
                   labels=['Simulated Prices', 'Realized Price', 'Conditional Price',
                           '4-Day In-Sample', '1-Day Out-of-Sample'],
                   loc='lower left')
            plt_trend = gcf()
            plt_trend.savefig(f'{folder}plt_{move_name}_{stock_name.lower()}.pdf', bbox_inches='tight')
            close()


if __name__ == '__main__':
    p2_1_trend()
