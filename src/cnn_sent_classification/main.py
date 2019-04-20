import os
import argparse
from exp_config import config
from models.model_utils import make_model
from learner import learn, predictions
from dataset import make_dataset, make_iterator
from utils import read_csv

dir = os.path.dirname(os.path.realpath(__file__))


def main(*args):
    full_search, dry_run, infer, model_name, best, result_dir, sub_fn_path = args

    if full_search:
        pass
    else:
        config['model_name'] = model_name
        config['result_dir'] = result_dir

        if infer:
            vocab, trn_ds, vld_ds, _, emb_matrix = make_dataset(config)
            trn_dl, vld_dl, _                    = make_iterator(config, vocab, trn_ds, vld_ds, _)
            
            config['vocab_size']                 = len(vocab.itos)
            config['emb_matrix']                 = emb_matrix
            
            model = make_model(config)
            model = learn(model, trn_dl, vld_dl, vocab, config)

        else:
            vocab, trn_ds, _, tst_ds, emb_matrix = make_dataset(config)
            trn_dl, _, tst_dl                    = make_iterator(config, vocab, trn_ds, _, tst_ds)
            
            config['vocab_size']                 = len(vocab.itos)
            config['emb_matrix']                 = emb_matrix

            model = make_model(config)
            model = learn(model, trn_dl, _, vocab, config)
            test_labels = read_csv(config['test_labels_path'])
            
            _  = predictions(model, tst_dl, None, test_labels, sub_fn_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Kaggle Jigsaw Toxic Comments Classification.')

    parser.add_argument('-fullsearch', type=bool, help='Perform a full search of hyper-parameters to find the best combination.')
    parser.add_argument('-dry_run', type=bool, help='Perform a dry run (testing purpose)')
    parser.add_argument('-infer', type=bool, help='Inference mode')
    parser.add_argument('-model_name', type=str, help='Unique name of the model.')
    parser.add_argument('-best', type=bool, help='Force to use the best known configuration.')
    parser.add_argument('-result_dir', type=str, help='Name of the directory to store/load model.')
    parser.add_argument('-sub_path', type=str, help='Full path of submission file.')

    args = parser.parse_args()

    full_search = args.fullsearch
    dry_run     = args.dry_run
    infer       = args.infer
    model_name  = args.model_name
    best        = args.best
    result_dir  = args.result_dir
    sub_fn_path = args.sub_path

    main(full_search, dry_run, infer, model_name, best, result_dir, sub_fn_path)
