from pandas import read_csv, DataFrame, merge
from numpy import log, mean, cov
from scipy.stats import multivariate_normal
from functools import reduce
from matplotlib.pyplot import figure, plot, legend, axhline, xlabel, ylabel, gcf, close
from pathlib import Path


def p3_3_risk_man():
    window_len = 257
    n_sims = 10000
    folder = './risk_management/'

    stock_px = read_csv('./raw/stock_px.csv', index_col=[0], parse_dates=[0])
    real_px = stock_px[window_len:]
    real_rets = log(real_px).diff().dropna()
    real_pf = mean(real_rets, 1)
    real_pf.name = 'REAL_PF'

    sim_pf = read_csv('./simulations/sim_pf.csv', index_col=[0], parse_dates=[0])
    sim_rets = log(sim_pf).diff().dropna()

    v_norm_90, v_norm_95, v_norm_99 = list(), list(), list()
    v_tgan_90, v_tgan_95, v_tgan_99 = list(), list(), list()
    window = 1
    while window < len(stock_px) - window_len:
        window_px = stock_px[window:window + window_len]
        window_rets = log(window_px).diff().dropna()
        mv_norm = multivariate_normal(mean(window_rets, 0), cov(window_rets, rowvar=False), allow_singular=True)
        sim_n_pfs = sorted(mean(mv_norm.rvs(n_sims, random_state=42), 1))

        v_norm_90.append(sim_n_pfs[int((1 - 0.90) * n_sims)])
        v_norm_95.append(sim_n_pfs[int((1 - 0.95) * n_sims)])
        v_norm_99.append(sim_n_pfs[int((1 - 0.99) * n_sims)])

        sim_t_pfs = sorted(sim_rets.iloc[window - 1])
        v_tgan_90.append(sim_t_pfs[int((1 - 0.90) * n_sims)])
        v_tgan_95.append(sim_t_pfs[int((1 - 0.95) * n_sims)])
        v_tgan_99.append(sim_t_pfs[int((1 - 0.99) * n_sims)])

        print(f'Risk Management Window: {window:4d}')
        window += 1

    var_list = [real_pf,
                DataFrame(v_norm_90, real_pf.index, columns=['V_NORM_90']),
                DataFrame(v_norm_95, real_pf.index, columns=['V_NORM_95']),
                DataFrame(v_norm_99, real_pf.index, columns=['V_NORM_99']),
                DataFrame(v_tgan_90, real_pf.index, columns=['V_TGAN_90']),
                DataFrame(v_tgan_95, real_pf.index, columns=['V_TGAN_95']),
                DataFrame(v_tgan_99, real_pf.index, columns=['V_TGAN_99'])]
    var_rets = reduce(lambda left, right: merge(left, right, on='Date'), var_list)

    figure(figsize=(6, 3))
    plot(real_pf, lw=0.3)
    plot(var_rets.V_NORM_90, lw=0.3, ls='-.', c='g')
    plot(var_rets.V_NORM_95, lw=0.3, ls='-.', c='y')
    plot(var_rets.V_NORM_99, lw=0.3, ls='-.', c='r')
    legend(['Actual Returns', '90%-VaR (Normal)', '95%-VaR (Normal)', '99%-VaR (Normal)'],
           prop={'size': 5})
    axhline(lw=0.5, c='k')
    xlabel('Time')
    ylabel('Returns')
    plt_var_n = gcf()
    close()

    figure(figsize=(6, 3))
    plot(real_pf, lw=0.3)
    plot(var_rets.V_TGAN_90, lw=0.3, c='g')
    plot(var_rets.V_TGAN_95, lw=0.3, c='y')
    plot(var_rets.V_TGAN_99, lw=0.3, c='r')
    legend(['Actual Returns', '90%-VaR (CTS-GAN)', '95%-VaR (CTS-GAN)', '99%-VaR (CTS-GAN)'],
           prop={'size': 5})
    axhline(lw=0.5, c='k')
    xlabel('Time')
    ylabel('Returns')
    plt_var_t = gcf()
    close()

    Path(f'{folder}').mkdir(parents=True, exist_ok=True)
    var_rets.to_csv(f'{folder}var.csv')
    round(DataFrame([[mean(real_pf < v_norm_90), mean(real_pf < v_tgan_90)],
                     [mean(real_pf < v_norm_95), mean(real_pf < v_tgan_95)],
                     [mean(real_pf < v_norm_99), mean(real_pf < v_tgan_99)]],
                    columns=['Normal', 'CTS-GAN'],
                    index=['90%', '95%', '99%']), 5).to_csv(f'{folder}var_e.csv')
    round(DataFrame([[mean((var_rets.V_NORM_90 - real_pf)[real_pf < v_norm_90]),
                      mean((var_rets.V_TGAN_90 - real_pf)[real_pf < v_tgan_90])],
                     [mean((var_rets.V_NORM_95 - real_pf)[real_pf < v_norm_95]),
                      mean((var_rets.V_TGAN_95 - real_pf)[real_pf < v_tgan_95])],
                     [mean((var_rets.V_NORM_99 - real_pf)[real_pf < v_norm_99]),
                      mean((var_rets.V_TGAN_99 - real_pf)[real_pf < v_tgan_99])]],
                    columns=['Normal', 'CTS-GAN'],
                    index=['90%', '95%', '99%']), 5).to_csv(f'{folder}var_e_delta.csv')
    round(DataFrame([[mean((real_pf - var_rets.V_NORM_90)[real_pf > v_norm_90]),
                      mean((real_pf - var_rets.V_TGAN_90)[real_pf > v_tgan_90])],
                     [mean((real_pf - var_rets.V_NORM_95)[real_pf > v_norm_95]),
                      mean((real_pf - var_rets.V_TGAN_95)[real_pf > v_tgan_95])],
                     [mean((real_pf - var_rets.V_NORM_99)[real_pf > v_norm_99]),
                      mean((real_pf - var_rets.V_TGAN_99)[real_pf > v_tgan_99])]],
                    columns=['Normal', 'CTS-GAN'],
                    index=['90%', '95%', '99%']), 5).to_csv(f'{folder}var_c_delta.csv')
    plt_var_n.savefig(f'{folder}plt_var_n.pdf', bbox_inches='tight')
    plt_var_t.savefig(f'{folder}plt_var_t.pdf', bbox_inches='tight')


if __name__ == '__main__':
    p3_3_risk_man()
