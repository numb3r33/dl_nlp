import os
import argparse
from exp_config import config
from models.model_utils import make_model
from learner import learn, predictions
from dataset import make_dataset, make_iterator
from utils import read_csv, load_model

dir = os.path.dirname(os.path.realpath(__file__))


def main(*args):
    path, csv, test_csv, test_labels, identifier, infer, model_name, result_dir, sub_fn_path, load = args

    config['path']        = path
    config['csv']         = csv
    config['test_csv']    = test_csv
    config['test_labels'] = test_labels
    config['model_name']  = model_name
    config['result_dir']  = result_dir
    config['identifier']  = identifier

    if infer:
        vocab, trn_ds, vld_ds, _, emb_matrix = make_dataset(config)
        trn_dl, vld_dl, _                    = make_iterator(config, vocab, trn_ds, vld_ds, _)
        
        config['vocab_size']                 = len(vocab.itos)
        
        model = make_model(config, emb_matrix)

        # load model from disk from previous iteration and just 
        if load:
            print('Loading model from disk from {}'.format(config['result_dir'] + config['model_name'] + '.pth'))

            model_dict = load_model(config['result_dir'] + config['model_name'] + '.pth')
            model      = model.load_state_dict(model_dict)
        else:
            model = learn(model, trn_dl, vld_dl, vocab, config)

    else:
        vocab, trn_ds, _, tst_ds, emb_matrix = make_dataset(config)
        trn_dl, _, tst_dl                    = make_iterator(config, vocab, trn_ds, _, tst_ds)
        
        config['vocab_size']                 = len(vocab.itos)
        config['emb_matrix']                 = emb_matrix

        model = make_model(config, emb_matrix)

        if load:
            print('Loading model from disk from {}'.format(config['result_dir'] + config['model_name'] + '_full.pth'))

            model_dict  = load_model(config['result_dir'] + config['model_name'] + '_full.pth')
            model       = model.load_state_dict(model_dict)
        else:
            model = learn(model, trn_dl, _, vocab, config)
        
        test_labels = read_csv(config['test_labels_path'])
        _  = predictions(model, tst_dl, None, test_labels, sub_fn_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Kaggle Jigsaw Toxic Comments Classification.')

    parser.add_argument('-path', type=str, help='Basepath where data is stored.')
    parser.add_argument('-csv',  type=str, help='Name of the training file.')
    parser.add_argument('-test_csv', type=str, help='Name of the test file.')
    parser.add_argument('-test_labels', type=str, help='Name of the test labels file.')
    parser.add_argument('-identifier', type=str, help='Timestamp identifier for models and config.')
    parser.add_argument('-infer', type=bool, help='Inference mode')
    parser.add_argument('-model_name', type=str, help='Unique name of the model.')
    parser.add_argument('-result_dir', type=str, help='Name of the directory to store/load model.')
    parser.add_argument('-sub_path', type=str, help='Full path of submission file.')
    parser.add_argument('-load', type=bool, help='Whether to load the model from disk or not.')

    args = parser.parse_args()

    path        = args.path
    csv         = args.csv
    test_csv    = None if args.test_csv == 'None' else args.test_csv
    test_labels = args.test_labels
    identifier  = args.identifier
    infer       = args.infer
    model_name  = args.model_name
    result_dir  = args.result_dir
    sub_fn_path = args.sub_path
    load        = args.load

    main(path, csv, test_csv, test_labels, identifier, infer, model_name, result_dir, sub_fn_path, load)
