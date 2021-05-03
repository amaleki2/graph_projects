import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='SDF Graph')

    # data parameters
    parser.add_argument('--data-folder', dest='data_folder', type=str, required=True,
                        help='folder where data are stored')
    parser.add_argument('--n-obj', dest='n_object', type=int, default=200,
                        help='number of training/test objects')
    parser.add_argument('--edge-method', dest='edge_method', type=str, default= 'proximity',
                        choices=['edge', 'proximity', 'both'],
                        help='graph edges method, options: `edge`=mesh edge, `proximity`=radial proximity, and `both`')
    parser.add_argument('--prox-radius', dest='prox_radius', type=float, default=0.2,
                        help='graph edges method, options: `edge`=mesh edge, `proximity`=radial proximity, and `both`')
    parser.add_argument('--min-n-edges', dest='min_n_edges', type=int, default=None,
                        help='minimum number of edges for each node, disregard if r > proximity radius.')
    parser.add_argument('--edge-feat-on', dest='edge_features_on', type=int, default=1, choices=[0, 1],
                        help='use edge features or not')
    parser.add_argument('--global-feat-on', dest='global_features_on', type=int, default=1, choices=[0, 1],
                        help='use global features or not')
    parser.add_argument('--include-reverse-edge', dest='include_reverse_edge', type=int, default=1, choices=[0, 1],
                        help='include reverse edge to make graph undirectional')
    parser.add_argument('--include-self-edge', dest='include_self_edge', type=int, default=1, choices=[0, 1],
                        help='include self edges')

    # model parameters
    parser.add_argument('--network-name', dest='network_name', type=str, required=True,
                        choices=['gat', 'gcn', 'epd'], help='network name, options: `gat`, `gcn`, `epd`')
    parser.add_argument('--last-layer-skip', dest='last_layer_skip', type=int, default=1, choices=[0, 1],
                        help='whether last layer should get a skip connection from the input or not')
    parser.add_argument('--n-hidden', dest='n_hidden', nargs='+', type=int, default=64,
                        help='number of neurons of hidden layers')
    parser.add_argument('--n-edge-in', dest='n_edge_in', type=int, default=3,
                        help='number of input edge features')
    parser.add_argument('--n-edge-out', dest='n_edge_out', type=int, default=1,
                        help='number of output edge features')
    parser.add_argument('--n-node-in', dest='n_node_in', type=int, default=3,
                        help='number of input node features')
    parser.add_argument('--n-node-out', dest='n_node_out', type=int, default=1,
                        help='number of output node features')
    parser.add_argument('--n-global-in', dest='n_global_in', type=int, default=3,
                        help='number of input global features')
    parser.add_argument('--n-global-out', dest='n_global_out', type=int, default=1,
                        help='number of output global features')
    parser.add_argument('--n-process', dest='n_process', type=int, default=5,
                        help='number of output global features')
    parser.add_argument('--weights-shared', dest='weights_shared', type=int, default=1, choices=[0, 1],
                        help='whether processors share weight or not')
    parser.add_argument('--full-output', dest='full_output', type=int, default=0, choices=[0, 1],
                        help='whether model returns all processors outputs')
    parser.add_argument('--head', dest='head', type=int, default=1,
                        help='number of heads in GAT')
    parser.add_argument('--negative-slope', dest='negative_slope', type=float, default=0.2,
                        help='number of output channels')

    # training parameters
    parser.add_argument('--n-epochs', dest='n_epochs', type=int, default=100,
                        help='number of epochs')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=5,
                        help='batch size')
    parser.add_argument('--save-every', dest='save_every', type=int, default=10,
                        help='save network every x epochs')
    parser.add_argument('--print-every', dest='print_every', type=int, default=25,
                        help='frequency of printing results')
    parser.add_argument('--save-name', dest='save_name', type=str, default="model",
                        help='save name')
    parser.add_argument('--lr', dest='lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--lr-gamma', dest='lr_gamma', type=float, default=0.2,
                        help='learning rate scheduler factor')
    parser.add_argument('--lr-step', dest='lr_step', type=int, default=250,
                        help='learning rate scheduler step size')
    parser.add_argument('--eval-frac', dest='eval_frac', type=float, default=0.1,
                        help='fraction of dataset for evaluation')
    return parser.parse_args()
