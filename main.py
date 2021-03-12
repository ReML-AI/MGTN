import argparse
from engine import *
from models import *
from coco import *
from util import *
from logger import *

parser = argparse.ArgumentParser(description='Graph Multi-Label Classification Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset (e.g. data/')
parser.add_argument('--image-size', '-i', default=448, type=int,
                    metavar='N', help='image size (default: 224)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--epoch_step', default=[30,40], type=int, nargs='+',
                    help='number of epochs to change learning rate')
parser.add_argument('--device_ids', default=[0], type=int, nargs='+',
                    help='number of epochs to change learning rate')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('-bt', '--batch-size-test', default=None, type=int,
                    metavar='N', help='mini-batch size for test (default: 16)')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lrd', '--learning-rate-decay', default=0.1, type=float,
                    metavar='LRD', help='learning rate decay')
parser.add_argument('--lrp', '--learning-rate-pretrained', default=0.1, type=float,
                    metavar='LR', help='learning rate for pre-trained layers')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=0, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--embedding', default='model/embedding/coco_glove_word2vec_80x300.pkl',
                    type=str, metavar='EMB', help='path to embedding (default: glove)')
parser.add_argument('--embedding-length', default=300, type=int, metavar='EMB',
                    help='embedding length (default: 300)')
parser.add_argument('--adj-file', default='model/topology/coco_adj.pkl', type=str, metavar='ADJ',
                    help='Adj file (default: model/topology/coco_adj.pkl')
parser.add_argument('--adj-strong-threshold', default=0.4, type=float, metavar='ADJTS',
                    help='Adj strong threshold  (default: 0.4)')
parser.add_argument('--adj-weak-threshold', default=0.2, type=float, metavar='ADJTW',
                    help='Adj weak threshold (default: 0.2)')
parser.add_argument('--mod-file', default='model/topology/coco_mod.txt', type=str, metavar='MOD',
                    help='Adj file (default: model/topology/coco_mod.txt')
parser.add_argument('--mlt', '--multi-learning-threshold', default=0.999, type=float, metavar='MLT',
                    help='Multi-learning threshold  (default: 0.999)')
parser.add_argument('--exp-name', dest='exp_name', default='coco', type=str, metavar='COCO',
                    help='Name of experiment to have different location to save checkpoints')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('-n', '--neptune', dest='neptune', action='store_true',
                    help='run with neptune')
parser.add_argument('--neptune-path', default='neptune.txt', type=str, metavar='PATH',
                    help='Neptune API keys (default: neptune.txt)')

def main():
    args = parser.parse_args()
    torch.backends.cudnn.benchmark = False
    init_log(args)

    train_dataset = COCO2014(args.data, phase="train", emb_name=args.embedding)
    val_dataset = COCO2014(args.data, phase="val", emb_name=args.embedding)
    num_classes = 80

    print('Embedding:', args.embedding, '(', args.embedding_length, ')')
    print('Adjacency file:', args.adj_file)
    print('Adjacency Strong Threshold:', args.adj_strong_threshold)
    print('Adjacency Weak Threshold:', args.adj_weak_threshold)
    print('Modularity file:', args.mod_file)

    if args.adj_strong_threshold < args.adj_weak_threshold:
        args.adj_weak_threshold = args.adj_strong_threshold

    model = mgtn_resnet(num_classes=num_classes,
                        t1=args.adj_strong_threshold,
                        t2=args.adj_weak_threshold,
                        adj_file=args.adj_file,
                        mod_file=args.mod_file,
                        emb_features=args.embedding_length,
                        ml_threshold=args.mlt)
    # define loss function (criterion)
    criterion = nn.MultiLabelSoftMarginLoss()

    # define optimizer
    optimizer = torch.optim.SGD(model.get_config_optim(args.lr, args.lrp), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    model_path = "checkpoint/coco/%s" % args.exp_name
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    state = {
        "batch_size": args.batch_size,
        "batch_size_test": args.batch_size if args.batch_size_test is None else args.batch_size_test,
        "image_size": args.image_size,
        "max_epochs": args.epochs,
        "evaluate": args.evaluate,
        "resume": args.resume,
        "num_classes": num_classes,
        "difficult_examples": True,
        "save_model_path": model_path,
        "workers": args.workers,
        "epoch_step": args.epoch_step,
        "lr": args.lr,
        "lr_decay": args.lrd,
        'mlt': args.mlt,
        "device_ids": args.device_ids,
        "neptune": args.neptune,
        "evaluate": True if args.evaluate else False
    }

    engine = GraphMultiLabelEngine(state)
    engine.learning(model, criterion, train_dataset, val_dataset, optimizer)


if __name__ == "__main__":
    main()
