from backend.models import (
    average_checkpoints,
    get_model,
)
import os
import torch
import numpy as np
import random
from types import SimpleNamespace
import yamlargparse
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def plot_embedding_and_attractors(
    emb,
    attractors,
    labels,
    save_path,
    n_speakers
):
    emb = emb.squeeze(0).detach().cpu().numpy()
    attractors = attractors.squeeze(0).detach().cpu().numpy()
    attractors = attractors[:n_speakers, :]

    scaler_e = StandardScaler()
    scaled_emb = scaler_e.fit_transform(emb)

    all_data = np.concatenate((scaled_emb, attractors), axis=0)

    pca = PCA(n_components=2)
    pca.fit(all_data)
    emb_2d = pca.transform(scaled_emb)
    attractors_2d = pca.transform(attractors)


    plt.figure(figsize=(10, 8))

    colors = ['blue', 'orange', 'green', 'red']
    labels_text = ['Silence', 'Spk 1', 'Spk 2', 'Overlap']
    for i in np.unique(labels):
        mask = labels == i
        plt.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                    label=labels_text[i], color=colors[i], alpha=0.7)
    plt.scatter(attractors_2d[:n_speakers, 0], attractors_2d[:n_speakers, 1],
                label='Attractors', color='mediumpurple', marker='X',
                s=300)
    plt.title('2D PCA of Embeddings and Attractors')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'embeddings.png'))
    plt.close()


def parse_arguments() -> SimpleNamespace:
    parser = yamlargparse.ArgumentParser(description='EEND inference')
    parser.add_argument('-c', '--config', help='config file path',
                        action=yamlargparse.ActionConfigFile)
    parser.add_argument('--context-size', default=0, type=int)
    parser.add_argument('--encoder-units', type=int,
                        help='number of units in the encoder')
    parser.add_argument('--epochs', type=str,
                        help='epochs to average separated by commas \
                        or - for intervals.')
    parser.add_argument('--feature-dim', type=int)
    parser.add_argument('--frame-size', type=int)
    parser.add_argument('--frame-shift', type=int)
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--hidden-size', type=int,
                        help='number of units in SA blocks')
    parser.add_argument('--infer-data-dir', help='inference data directory.')
    parser.add_argument('--input-transform', default='',
                        choices=['logmel', 'logmel_meannorm',
                                 'logmel_meanvarnorm'],
                        help='input normalization transform')
    parser.add_argument('--log-report-batches-num', default=1, type=float)
    parser.add_argument('--median-window-length', default=11, type=int)
    parser.add_argument('--model-type', default='TransformerEDA',
                        help='Type of model (for now only TransformerEDA)')
    parser.add_argument('--models-path', type=str,
                        help='directory with model(s) to evaluate')
    parser.add_argument('--num-frames', default=-1, type=int,
                        help='number of frames in one utterance')
    parser.add_argument('--num-speakers', type=int)
    parser.add_argument('--rttms-dir', type=str,
                        help='output directory for rttm files.')
    parser.add_argument('--sampling-rate', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--subsampling', default=10, type=int)
    parser.add_argument('--threshold', default=0.5, type=float)
    parser.add_argument('--transformer-encoder-n-heads', type=int)
    parser.add_argument('--transformer-encoder-n-layers', type=int)
    parser.add_argument('--transformer-encoder-dropout', type=float)
    parser.add_argument('--vad-loss-weight', default=0.0, type=float)

    attractor_args = parser.add_argument_group('attractor')
    attractor_args.add_argument(
        '--time-shuffle', action='store_true',
        help='Shuffle time-axis order before input to the network')
    attractor_args.add_argument('--attractor-loss-ratio', default=1.0,
                                type=float, help='weighting parameter')
    attractor_args.add_argument('--attractor-encoder-dropout',
                                default=0.1, type=float)
    attractor_args.add_argument('--attractor-decoder-dropout',
                                default=0.1, type=float)
    attractor_args.add_argument('--estimate-spk-qty', default=-1, type=int)
    attractor_args.add_argument('--estimate-spk-qty-thr',
                                default=-1, type=float)
    attractor_args.add_argument(
        '--detach-attractor-loss', default=False, type=bool,
        help='If True, avoid backpropagation on attractor loss')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # 1. Parse arguments
    args = parse_arguments()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    np.random.seed(args.seed)  # Numpy module.
    random.seed(args.seed)  # Python random module.
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    args.device = torch.device('cpu')

    # 2. Load model
    model = get_model(args)
    args.models_path = '/mnt/impress/Lab/EEND/exp/EEND_EDA/simu/models'
    model = average_checkpoints(
        args.device, model, args.models_path, args.epochs)
    model.eval()

    # 3. Load data
    data_path = '/mnt/impress/Lab/EEND/DB'
    data_type = 'simu'
    data_tdt = 'test_all_ns2_beta5_500'
    data_name = os.listdir(os.path.join(data_path, data_type, data_tdt, '.cache'))[420]

    data = np.load(os.path.join(data_path, data_type, data_tdt, '.cache', data_name))

    y = torch.from_numpy(data['Y']).unsqueeze(0)
    t = torch.from_numpy(data['T'])

    n_speakers = args.estimate_spk_qty
    powers = torch.tensor(torch.arange(1, n_speakers + 1))
    labels_1d = torch.sum(t * powers, dim=-1)

    # 4. Inference
    emb = model.get_embeddings(y)
    if args.time_shuffle:
        orders = [np.arange(e.shape[0]) for e in emb]
        for order in orders:
            np.random.shuffle(order)
        attractors, probs = model.eda.estimate(
            torch.stack([e[order] for e, order in zip(emb, orders)]))
    else:
        attractors, probs = model.eda.estimate(emb)
    
    # 5. Plot embeddings and attractors
    print(f'---> Load data from {data_name}')
    save_path = '/mnt/impress/Lab/EEND/exp/EEND_EDA/simu/img'
    plot_embedding_and_attractors(emb,
                                  attractors,
                                  labels_1d,
                                  save_path,
                                  n_speakers = args.estimate_spk_qty)