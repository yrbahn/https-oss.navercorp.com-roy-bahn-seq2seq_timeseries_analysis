import argparse
import tensorflow as tf
from dataset import TimeSeriesDataset
from seq2seq_model import Seq2SeqModel

tf.logging.set_verbosity(tf.logging.INFO)


def train():
    parser = argparse.ArgumentParser(description='train model based on data over the period')
    parser.add_argument('--train-data-files', required=True, type=str,
                       help='train data files')
    parser.add_argument('--eval-data-files', type=str,
                        help='eval data files')
    parser.add_argument('--batch-size', type=int,
                        default=512, help='batch size')
    parser.add_argument('--hidden-size', type=int,
                        default=64, help='hidden size in rnn')
    parser.add_argument('--input-dim', type=int,
                        default=1, help='input data dimension')
    parser.add_argument('--output-dim', type=int,
                        default=1, help='output data dimension')
    parser.add_argument('--input-seq-len', type=int,
                        default=10, help='input sequnece length')
    parser.add_argument('--output-seq-len', type=int,
                        default=5, help='output sequence length')
    parser.add_argument('--optimizer', type=str,
                        default='Adam', help='optimizer')
    parser.add_argument('--learning_rate', type=float,
                        default=0.01, help='learning rate')
    parser.add_argument('--layer_cnt', type=int,
                        default=1, help='seq2seq layer cnt')
    parser.add_argument('--lambda-l2-reg', type=float,
                        default=0.02, help='lambda l2 reg')
    parser.add_argument('--gradient_clipping', type=float,
                        default=2.5, help='gradient clippling')
    parser.add_argument('--model_dir', type=str,
                        required=True, help='model output directory')
    parser.add_argument('--steps', type=int,
                        default=None, help='training steps')
    parser.add_argument('--epoch', type=int,
                        default=None, help='training epoch')

    args = parser.parse_args()
    
    def create_params():
        return {'optimizer' : args.optimizer,
                'learning_rate' : args.learning_rate,
                'lambda_l2_reg' : args.lambda_l2_reg,
                'gradient_clipping': args.gradient_clipping,
                'steps': args.steps}

    hparams = create_params()

    seq2seq_model = Seq2SeqModel(args.input_dim, args.output_dim,
                                 args.input_seq_len, args.output_seq_len,
                                 args.hidden_size, args.layer_cnt)

    seq2seq_model_fn = seq2seq_model.create_model_fn()

    estimator = tf.estimator.Estimator(model_fn=seq2seq_model_fn,
                                       model_dir=args.model_dir,
                                       params=hparams)

    #train input fn
    train_dataset = TimeSeriesDataset(args.train_data_files,
                                      args.epoch,
                                      args.batch_size,
                                      args.input_seq_len,
                                      args.output_seq_len)

    estimator.train(input_fn=train_dataset.input_fn,
                    steps=args.steps)

    eval_dataset = TimeSeriesDataset(args.eval_data_files,
                                     1,
                                     args.batch_size,
                                     args.input_seq_len,
                                     args.output_seq_len)

    estimator.evaluate(input_fn=eval_dataset.input_fn)

    
if __name__ == "__main__":
    train()
