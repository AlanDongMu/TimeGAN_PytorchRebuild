import argparse
import numpy as np

# 1. Training model
from run import train, test
# 2. Data loading
from data_loading import real_data_loading, sine_data_generation


def main(opt):
    # Data loading
    ori_data = None
    if opt.data_name in ['stock', 'energy']:
        ori_data = real_data_loading(opt.data_dir, opt.data_name, opt.seq_len)
    elif opt.data_name == 'sine':
        # Set number of samples and its dimensions
        ori_data = sine_data_generation(opt.sine_no, opt.seq_len, opt.sine_dim)

    print(opt.data_name + ' dataset is ready.')

    # Training or Testing
    if opt.is_test:
        test(opt, ori_data)
    else:
        train(opt, ori_data)
        test(opt, ori_data)


if __name__ == '__main__':
    """Main function for timeGAN experiments.
    Args:
      - data_name: sine, stock, or energy
      - seq_len: sequence length
      - Network parameters (should be optimized for different datasets)
        - module: gru, lstm
        - hidden_dim: hidden dimensions
        - num_layer: number of layers
        - iteration: number of training iterations
        - batch_size: the number of samples in each batch
      - metric_iteration: number of iterations for metric computation
    Returns:
      - ori_data: original data
      - gen_data: generated synthetic data
      - metric_results: discriminative and predictive scores
    """
    # Args for the main function
    parser = argparse.ArgumentParser()
    # Data parameters
    parser.add_argument('--data_name', type=str, default='stock', choices=['sine', 'stock', 'energy'], )
    parser.add_argument('--seq_len', type=int, default=24, help='sequence length')
    parser.add_argument('--sine_no', type=int, default=10000, help='number of sine data samples')
    parser.add_argument('--sine_dim', type=int, default=5, help='dim of  sine data')
    # Network parameters (should be optimized for different datasets)
    parser.add_argument('--module', choices=['gru', 'lstm'], default='gru', type=str)
    parser.add_argument('--hidden_dim', type=int, default=24, help='hidden state dimensions')
    parser.add_argument('--num_layer', type=int, default=3, help='number of layers')
    # Model training and testing parameters
    parser.add_argument('--gamma', type=float, default=1, help='gamma weight for G_loss and D_loss')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for adam')
    parser.add_argument('--iterations', type=int, default=50000, help='Training iterations')
    parser.add_argument('--print_times', type=int, default=10, help='Print times when Training')
    parser.add_argument('--batch_size', type=int, default=128, help='the number of samples in mini-batch')
    parser.add_argument('--synth_size', type=int, default=0, help='the number of samples in synthetic data, '
                                                                  '0--len(ori_data)')
    parser.add_argument('--metric_iteration', type=int, default=10, help='iterations of the metric computation')
    # Save and Load
    parser.add_argument('--data_dir', type=str, default="./data", help='path to stock and energy data')
    parser.add_argument('--networks_dir', type=str, default="./trained_networks", help='path to checkpoint')
    parser.add_argument('--output_dir', type=str, default="./output", help='folder to output metrics and images')
    # Model running parameters
    parser.add_argument('--is_test', type=bool, default=False, help='iterations of the metric computation')
    parser.add_argument('--only_visualize_metric', type=bool, default=False, help='only compute visualization metrics')
    parser.add_argument('--load_checkpoint', type=bool, default=False, help='load pretrain networks')

    # Call main function
    opt = parser.parse_args()
    main(opt)
