from keras.utils import set_random_seed
from tensorflow import config, convert_to_tensor
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from pandas import read_csv, DataFrame
from p1_0_utils import *
from keras.optimizers import Adam
from tensorflow.data import Dataset
from tensorflow import GradientTape, sqrt, reduce_mean, square
from tensorflow import reshape, tile, concat, nn, abs
from tensorflow.math import log


def p1_1_cts_gan():
    learn_rate = 2e-4
    sequence_len = 12
    condition_len = 7
    batch_size = 32
    window_len = 257
    n_itts = 2000
    n_gen_itts = 2
    folder = './losses/'

    set_random_seed(42)
    config.experimental.enable_op_determinism()
    Path('./models/').mkdir(parents=True, exist_ok=True)
    mms = MinMaxScaler()

    stock_px = read_csv('./raw/stock_px.csv', index_col=[0], parse_dates=[0])
    stock_len, f_dim = stock_px.shape

    units = f_dim * 4
    e_net = make_net(3, units, units, 'embedder')
    r_net = make_net(3, units, f_dim, 'recovery')
    s_net = make_net(2, units, units, 'supervisor')
    g_net = make_net(2, units, units, 'generator')
    d_net = make_net(2, units, 1, 'discriminator')

    r_optm = Adam(2 * learn_rate)
    s_optm = Adam(2 * learn_rate)
    g_optm = Adam(1 * learn_rate)
    a_optm = Adam(1 * learn_rate)
    d_optm = Adam(3 * learn_rate)

    gen_len = sequence_len - condition_len
    z_iter = iter(Dataset.from_generator(gen_z, float32, args=(batch_size, gen_len, f_dim)).repeat())

    report_t1, report_t2, report_t3 = list(), list(), list()
    window = 0
    while window < stock_len - window_len:
        window_px = stock_px[window:window + window_len]
        window_diff = window_px.diff().dropna()
        scaled_diff = mms.fit_transform(window_diff)
        scaled_data = convert_to_tensor(scaled_diff, float32)

        data_len = window_len - sequence_len
        data_list = [scaled_data[i:i + sequence_len] for i in range(data_len)]
        data_iter = iter(Dataset.from_tensor_slices(data_list)
                         .shuffle(data_len).batch(batch_size, True).repeat())

        # Train the embedder and recovery networks:
        for itt in range(n_itts):
            r = next(data_iter)
            x = r[:, condition_len:, :]
            with GradientTape() as tape:
                h = e_net(x)
                x_tilde = r_net(h)

                r_loss = sqrt(reduce_mean(square(x - x_tilde)))
            r_vars = e_net.trainable_variables + r_net.trainable_variables
            r_grad = tape.gradient(r_loss, r_vars)
            r_optm.apply_gradients(zip(r_grad, r_vars))

            r_loss = r_loss.numpy()
            report_t1.append({'WINDOW': window, 'ITERATION': itt, 'R_LOSS': r_loss})
            if itt % 100 == 0:
                print(f'T1 || W: {window:4d} | I: {itt:4d} | R: {r_loss:.4f}')

        # Train the supervisor network:
        for itt in range(n_itts):
            r = next(data_iter)
            x = r[:, condition_len:, :]
            with GradientTape() as tape:
                h = e_net(x)
                h_s = s_net(h)

                s_loss = sqrt(reduce_mean(square(h - h_s)))
            s_vars = s_net.trainable_variables
            s_grad = tape.gradient(s_loss, s_vars)
            s_optm.apply_gradients(zip(s_grad, s_vars))

            s_loss = s_loss.numpy()
            report_t2.append({'WINDOW': window, 'ITERATION': itt, 'S_LOSS': s_loss})
            if itt % 100 == 0:
                print(f'T2 || W: {window:4d} | I: {itt:4d} | S: {s_loss:.4f}')

        g_loss = a_loss = 0
        for itt in range(n_itts):
            for _ in range(n_gen_itts):
                r = next(data_iter)
                c = r[:, :condition_len, :]
                x = r[:, condition_len:, :]

                z = next(z_iter)
                c = reshape(c, (batch_size, -1))
                c = reshape(tile(c, (1, gen_len)), (batch_size, gen_len, -1))
                cz = concat((c, z), -1)

                # Train the generator and supervisor networks:
                with GradientTape() as tape:
                    h_hat = g_net(cz)
                    ch_hat = concat((c, h_hat), -1)
                    g_loss_fake_u = -reduce_mean(log(d_net(ch_hat)))

                    h_hat_s = s_net(h_hat)
                    ch_hat_s = concat((c, h_hat_s), -1)
                    g_loss_fake_s = -reduce_mean(log(d_net(ch_hat_s)))

                    h = e_net(x)
                    h_s = s_net(h)
                    g_s_loss = sqrt(reduce_mean(square(h - h_s)))

                    x_hat = r_net(h_hat_s)
                    x_m, x_var = nn.moments(x, 0)
                    x_hat_m, x_hat_var = nn.moments(x_hat, 0)
                    g_loss_m = reduce_mean(abs(x_m - x_hat_m))
                    g_loss_sd = reduce_mean(abs(sqrt(x_var) - sqrt(x_hat_var)))

                    g_loss = g_loss_fake_u + g_loss_fake_s + 100 * g_s_loss + 100 * (g_loss_m + g_loss_sd)
                g_vars = g_net.trainable_variables + s_net.trainable_variables
                g_grad = tape.gradient(g_loss, g_vars)
                g_optm.apply_gradients(zip(g_grad, g_vars))

                # Train the embedder and recovery networks:
                with GradientTape() as tape:
                    h = e_net(x)
                    x_tilde = r_net(h)
                    r_loss = sqrt(reduce_mean(square(x - x_tilde)))

                    h_s = s_net(h)
                    s_loss = sqrt(reduce_mean(square(h - h_s)))

                    a_loss = 10 * r_loss + 0.1 * s_loss
                a_vars = e_net.trainable_variables + r_net.trainable_variables
                a_grad = tape.gradient(a_loss, a_vars)
                a_optm.apply_gradients(zip(a_grad, a_vars))

            # Train the discriminator network:
            r = next(data_iter)
            c = r[:, :condition_len, :]
            x = r[:, condition_len:, :]

            z = next(z_iter)
            c = reshape(c, (batch_size, -1))
            c = reshape(tile(c, (1, gen_len)), (batch_size, gen_len, -1))
            cz = concat((c, z), -1)

            h = e_net(x)
            ch = concat((c, h), -1)

            h_s = s_net(h)
            ch_s = concat((c, h_s), -1)

            h_hat = g_net(cz)
            ch_hat = concat((c, h_hat), -1)

            h_hat_s = s_net(h_hat)
            ch_hat_s = concat((c, h_hat_s), -1)

            with GradientTape() as tape:
                d_loss_real_u = -reduce_mean(log(d_net(ch)))
                d_loss_real_s = -reduce_mean(log(d_net(ch_s)))
                d_loss_fake_u = -reduce_mean(log(1 - d_net(ch_hat)))
                d_loss_fake_s = -reduce_mean(log(1 - d_net(ch_hat_s)))

                d_loss = d_loss_real_u + d_loss_fake_u + d_loss_real_s + d_loss_fake_s
            d_vars = d_net.trainable_variables
            d_grad = tape.gradient(d_loss, d_vars)
            d_optm.apply_gradients(zip(d_grad, d_vars))

            d_loss = d_loss.numpy()
            g_loss = g_loss.numpy()
            a_loss = a_loss.numpy()
            report_t3.append({'WINDOW': window, 'ITERATION': itt, 'D_LOSS': d_loss, 'G_LOSS': g_loss, 'A_LOSS': a_loss})
            if itt % 100 == 0:
                print(f'T3 || W: {window:4d} | I: {itt:4d} | D: {d_loss:.4f} | G: {g_loss:.4f} | A: {a_loss:.4f}')

        r_net.save_weights(f'./models/w{window}_r.h5')
        s_net.save_weights(f'./models/w{window}_s.h5')
        g_net.save_weights(f'./models/w{window}_g.h5')

        n_itts = 10
        window += 1

    Path(folder).mkdir(parents=True, exist_ok=True)
    DataFrame(report_t1).to_csv(f'{folder}loss_report_t1.csv', index=False)
    DataFrame(report_t2).to_csv(f'{folder}loss_report_t2.csv', index=False)
    DataFrame(report_t3).to_csv(f'{folder}loss_report_t3.csv', index=False)


if __name__ == '__main__':
    p1_1_cts_gan()
