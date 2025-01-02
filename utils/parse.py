def parser_add_main_args(parser):
    parser.add_argument('--dataset', type=str, default='Deng', help='Path of data for predicting.')
    parser.add_argument('--data_dir', type=str, default='../data/')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--eval_step', type=int,
                        default=1, help='how often to print')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--runs', type=int, default=1,
                        help='number of distinct runs')
    parser.add_argument('--train_prop', type=float, default=.6,
                        help='training label proportion')
    parser.add_argument('--valid_prop', type=float, default=.2,
                        help='validation label proportion')
    parser.add_argument('--metric', type=str, default='acc', choices=['acc', 'rocauc', 'f1'],
                        help='evaluation metric')
    parser.add_argument('--knn_num', type=int, default=5, help='number of k for KNN graph')
    parser.add_argument('--T', type=float, default=1.0, help='temperature for Softmax')

    # hyper-parameter for model arch and training
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-3)
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers for deep methods')
    parser.add_argument('--num_mlp_layers', type=int, default=1,
                        help='number of mlp layers')    
    parser.add_argument('--use_bn', action='store_true', help='use layernorm')
    parser.add_argument('--m_in', type=float, default=-5, help='upper bound for in-distribution energy')
    parser.add_argument('--m_out', type=float, default=-1, help='lower bound for in-distribution energy')
    parser.add_argument('--use_prop', action='store_true', help='whether to use energy belief propagation')
    parser.add_argument('--oodprop', type=int, default=2, help='number of layers for energy belief propagation')
    parser.add_argument('--oodalpha', type=float, default=0.3, help='weight for residual connection in propagation')
    parser.add_argument('--use_zinb', action='store_true', help='whether to use ZINB loss (use if you do not need this)')
    parser.add_argument('--use_2hop', action='store_false', help='whether to use 2-hop propagation (use if you do not need this)')
    parser.add_argument('--zinb_weight', type=float, default=1e-4)

    # display and utility
    parser.add_argument('--display_step', type=int,
                        default=10, help='how often to print')
    parser.add_argument('--print_prop', action='store_true',
                        help='print proportions of predicted class')
    parser.add_argument('--print_args', action='store_true',
                        help='print args for hyper-parameter searching')
    parser.add_argument('--cl_weight', type=float, default=0.0)
    parser.add_argument('--mask_ratio', type=float, default=0.8)
    parser.add_argument('--spatial', action='store_false', help='read spatial')
