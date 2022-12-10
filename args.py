import argparse
from argparse import Namespace

def get_args():
    parser = argparse.ArgumentParser(description='Style Transfer.')
    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--batch_size_test', default=32, type=int)

    parser.add_argument('--pretrained_model', type=str, default='facebook/bart-base')
    parser.add_argument('--pretrain_step', default=2, type=int)

    parser.add_argument('--load_model', type=str, default='')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--Epoch', type=int, default=50) 
    parser.add_argument('--eval_steps', type=int, default=50)

    parser.add_argument('--max_length', type=int, default=320)
    parser.add_argument('--length_title', type=int, default=75)
    parser.add_argument('--length_p_v', type=int, default=100)
    parser.add_argument('--length_maidian', type=int, default=200)

    # parser.add_argument('--length_p_v', type=int, default=120)
    # parser.add_argument('--length_maidian', type=int, default=220)

    parser.add_argument('--save_path', type=str, default='./save/')
    parser.add_argument('--factor_aug', type=float, default=0.01)
    parser.add_argument('--factor_plan', type=float, default=1.0)
    parser.add_argument('--factor_style', type=float, default=1.0)

    parser.add_argument('--pseudo_sample', type=bool, default=False)
    parser.add_argument('--planning', type=bool, default=False)
    parser.add_argument('--style_disentange', type=bool, default=False)
    parser.add_argument('--graph', type=bool, default=False)

    parser.add_argument('--parallel', default=False, type=bool, help="Weather to use multi-gpus.")
    parser.add_argument('--train_scrach', type=bool, default=False)

    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--log_file', type=str, default='log')
    parser.add_argument('--model_name', type=str, default='ours')

    parser.add_argument('--num_nodes', type=int, default=30)
    parser.add_argument('--teacher_force', type=float, default=1.0)

    args = parser.parse_args()
    return args
