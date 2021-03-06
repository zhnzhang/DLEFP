import argparse


def get_train_args():
    parser=argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str, default="../checkpoint/model")
    parser.add_argument('--output_path', type=str, default="../result/output")

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--test_batch_size', type=int, default=20)
    parser.add_argument('--nepoch', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--gpu', type=bool, default=True, help='是否使用gpu')
    parser.add_argument('--num_workers', type=int, default=2, help='dataloader使用的线程数量')

    parser.add_argument('--labmda', type=float, default=0.2)

    # gcn
    parser.add_argument('--gcn_layers', type=int, default=2)
    parser.add_argument('--gcn_hid_dim', type=int, default=768)
    parser.add_argument('--gcn_out_dim', type=int, default=768)

    parser.add_argument('--activation', type=str, default="tanh")
    parser.add_argument('--dropout', type=float, default=0)
    # parser.add_argument('--linear_dim', type=int, default=300)

    # parser.add_argument('--type_embed_dim', type=int, default=0)

    # BERT
    parser.add_argument('--bert_hid_size', type=int, default=768)
    parser.add_argument('--bert_path', type=str, default="bert-base-uncased")
    parser.add_argument('--bert_fix', default=False, action='store_true')

    parser.add_argument('--n_heads', type=int, default=8)

    parser.add_argument('--data_path', type=str, default='../../../data/dlef_corpus/english.xml')
    parser.add_argument('--data_save_path', type=str, default='../english.pkl')

    parser.add_argument('--k_fold', type=int, default=10)

    opt = parser.parse_args()
    print(opt)
    return opt
