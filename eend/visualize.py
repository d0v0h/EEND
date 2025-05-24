import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import yamlargparse
from types import SimpleNamespace


def plot_embedding_with_centers(
    emb,           # (T, F) torch.Tensor
    labels,        # (T,) numpy array or torch.Tensor, 0: silence, 1: spk1, 2: spk2, 3: overlap 등
    centers=None,  # (C, F) torch.Tensor or None
    attractors=None, # (C, F) torch.Tensor or None
    save_path=None,
    title='Embedding & Attractor/Center PCA', # 기본값도 PCA로 변경하는 것이 좋습니다.
    n_speakers=None
):    
    frames = emb.shape[0]

    # Convert to numpy array
    emb_np = emb.cpu().numpy()
    if attractors is not None:
        attr_np = attractors.cpu().numpy()
        all_data_np = np.concatenate((emb_np, attr_np), axis=0)
    elif centers is not None:
        centers_np = centers.cpu().numpy()
        all_data_np = np.concatenate((emb_np, centers_np), axis=0)
    
    print(emb_np.mean(), emb_np.std())
    print(attr_np.mean(), attr_np.std())
    print(all_data_np.mean(), all_data_np.std())

    # Normalization & PCA
    scaler = StandardScaler()
    pca = PCA(n_components=2)

    all_data_scaled = scaler.fit_transform(all_data_np)

    print(all_data_scaled.shape)

    print(all_data_scaled.mean(), all_data_scaled.std())
    print(all_data_scaled[:-2:].mean(), all_data_scaled[:-2].std())
    print(all_data_scaled[-2:].mean(), all_data_scaled[-2:].std())
    print(emb_np.max(), emb_np.min())
    print(attr_np.max(), attr_np.min())

    all_data_pca = pca.fit_transform(all_data_scaled)

    emb_pca = all_data_pca[:frames, :]
    if attractors is not None:
        other_pca = all_data_pca[frames:, :]
        anotation = 'Attractor'
    elif centers is not None:
        other_pca = all_data_pca[frames:, :]
        anotation = 'Center'
    
    # Plotting
    plt.figure(figsize=(10, 8))

    # Plot embeddings
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    labels_text = ['Silence', 'Spk 1', 'Spk 2', 'Overlap']
    for i in np.unique(labels):
        mask = labels == i
        plt.scatter(emb_pca[mask, 0], emb_pca[mask, 1],
                    label = labels_text[i],
                    color =colors[i],
                    alpha=0.6
        )
    
    # Plot centers or attractors
    plt.scatter(
        other_pca[:, 0], other_pca[:, 1],
        marker='X', color='mediumpurple', label=anotation,
        s=300
    )
    
    plt.title(title)
    plt.legend()

    plt.savefig(save_path)
    plt.close()


def parse_arguments() -> SimpleNamespace:
    parser = yamlargparse.ArgumentParser(description='EEND inference')
    parser.add_argument('--type', type=str, default='EDA',
                        choices=['EDA', 'CA'],
                        help='Type of model to use (EDA or CA)')
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
    args = parse_arguments()

    # For reproducibility
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

    # Setting epochs
    data_type = 'simu'
    init_epochs = '90-100'

    # Setting EXP type
    if args.type == 'EDA':
        from backend.models import get_model, average_checkpoints
        model_path = f'/mnt/impress/Lab/EEND/exp/EEND_EDA/{data_type}/models'
        model = get_model(args)
        model = average_checkpoints(
            args.device, model, model_path, init_epochs)
    elif args.type == 'CA':
        from backend.models_ca import get_model, average_checkpoints
        model_path = f'/mnt/impress/Lab/EEND/exp/EEND_CA/{data_type}/models'
        model = get_model(args)
        model = average_checkpoints(
            args.device, model, model_path, init_epochs)
    else:
        raise ValueError(f"Unknown type: {args.type}")
    model.eval()


    # Load data
    data_path = f'/mnt/impress/Lab/EEND/DB/{data_type}/test_all_ns2_beta5_500/.cache/data_simu_wav_test_all_ns2_beta5_500_1_mix_0000001_0_19660.npz'
    data = np.load(data_path)

    feature = data['Y']                     # (T, F)
    label = data['T']                       # (T, C)
    feature = torch.from_numpy(feature).unsqueeze(0).to(args.device)
    label = torch.from_numpy(label).to(args.device)

    n_speakers = label.shape[1]

    powers = torch.tensor(torch.arange(1, n_speakers + 1), device=args.device)
    labels_1d = torch.sum(label * powers, dim=-1)

    save_path = f'/mnt/impress/Lab/EEND/exp/EEND_{args.type}/{data_type}/img/embeddings.png'
    
    print(f'speakers: {n_speakers}')
    with torch.no_grad():
        emb = model.enc(feature)                # (T, D)

        if args.type == 'EDA':
            emb_temp = emb.unsqueeze(0)
            _, attractors = model.eda(emb_temp, [n_speakers])
            attractors = attractors.squeeze(0)  # (C, D)

            plot_embedding_with_centers(
                emb=emb,
                labels=labels_1d,
                attractors=attractors,
                save_path=save_path,
                title='Embedding & Attractor PCA',
                n_speakers=n_speakers
            )

        elif args.type == 'CA':
            centers = model.ca.centers.weight
            centers = centers[:n_speakers, :]
        
            plot_embedding_with_centers(
                emb=emb,
                labels=labels_1d,
                centers=centers,
                save_path=save_path,
                title='Embedding & Center PCA',
                n_speakers=n_speakers
            )