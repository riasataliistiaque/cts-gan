from keras.utils import set_random_seed
from tensorflow import convert_to_tensor, float32, reduce_mean
from pathlib import Path
from pandas import read_csv
from tensorflow.data import Dataset
from numpy import concatenate
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib.pyplot import figure, scatter, legend, xticks, yticks, gcf, close
from numpy import log


def p3_0_pca_tsne():
    folder = './visualizations/'
    window_len = 257
    sequence_len = 12
    batch_size = 2000

    set_random_seed(42)
    Path(folder).mkdir(parents=True, exist_ok=True)

    stock_px = read_csv('./raw/stock_px.csv', index_col=[0], parse_dates=[0])
    window_px = stock_px[window_len:]
    window_rets = log(window_px).diff().dropna()

    window_px = convert_to_tensor(window_px, float32)
    window_rets = convert_to_tensor(window_rets, float32)

    data_len = len(window_px)
    data_list = [window_px[i:i + sequence_len] for i in range(data_len - sequence_len)]
    data_iter = iter(Dataset.from_tensor_slices(data_list)
                     .shuffle(data_len).batch(batch_size, True).repeat())
    r_px = next(data_iter)

    data_len = len(window_rets)
    data_list = [window_rets[i:i + sequence_len] for i in range(data_len - sequence_len)]
    data_iter = iter(Dataset.from_tensor_slices(data_list)
                     .shuffle(data_len).batch(batch_size, True).repeat())
    r_rets = next(data_iter)

    sim_px = read_csv('./simulations/sim_px.csv', index_col=[0], parse_dates=[0])
    sim_rets = log(sim_px).diff().dropna()

    sim_px = convert_to_tensor(sim_px, float32)
    sim_rets = convert_to_tensor(sim_rets, float32)

    sim_len = len(sim_px)
    sim_list = [sim_px[i:i + sequence_len] for i in range(sim_len - sequence_len)]
    sim_iter = iter(Dataset.from_tensor_slices(sim_list)
                    .shuffle(sim_len).batch(batch_size, True).repeat())
    f_px = next(sim_iter)

    sim_len = len(sim_rets)
    sim_list = [sim_rets[i:i + sequence_len] for i in range(sim_len - sequence_len)]
    sim_iter = iter(Dataset.from_tensor_slices(sim_list)
                    .shuffle(sim_len).batch(batch_size, True).repeat())
    f_rets = next(sim_iter)

    r_px = reduce_mean(r_px, -1).numpy()
    f_px = reduce_mean(f_px, -1).numpy()
    rf_px = concatenate((r_px, f_px), 0)

    r_rets = reduce_mean(r_rets, -1).numpy()
    f_rets = reduce_mean(f_rets, -1).numpy()
    rf_rets = concatenate((r_rets, f_rets), 0)

    tsne = TSNE(n_iter=50000, verbose=2, random_state=42)
    tsne_data = tsne.fit_transform(rf_px)

    pca = PCA(2)
    pca_data = pca.fit_transform(rf_rets)

    figure(figsize=(8, 6))
    scatter(pca_data[:batch_size, 0], pca_data[:batch_size, 1], c='blue')
    scatter(pca_data[batch_size:, 0], pca_data[batch_size:, 1], c='orange', marker='x', s=5)
    legend(labels=['Real', 'CTS-GAN'])
    xticks([])
    yticks([])
    plt_pca = gcf()
    close()

    figure(figsize=(8, 6))
    scatter(tsne_data[:batch_size, 0], tsne_data[:batch_size, 1], c='blue')
    scatter(tsne_data[batch_size:, 0], tsne_data[batch_size:, 1], c='orange', marker='x', s=5)
    legend(labels=['Real', 'CTS-GAN'])
    xticks([])
    yticks([])
    plt_tsne = gcf()
    close()

    plt_pca.savefig(f'{folder}plt_pca.pdf', bbox_inches='tight')
    plt_tsne.savefig(f'{folder}plt_tsne.pdf', bbox_inches='tight')


if __name__ == '__main__':
    p3_0_pca_tsne()
