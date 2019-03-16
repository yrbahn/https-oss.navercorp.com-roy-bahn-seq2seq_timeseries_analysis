import tensorflow as tf

COLUMN_NAMES = ["index", "dow_0", "dow_1", "value"]
COLUMN_DEFAULTS = [[""], [0.0], [0.0], [0.0]]


class TimeSeriesDataset:
    def __init__(self, csv_filename, epoch=None, batch_size=128, input_seq_len=10,
                 output_seq_len=5, stride=2, prefetch_buffer_size=100, csv_delimiter=','):
        self.epoch = epoch
        self.batch_size = batch_size
        self.csv_delimiter = csv_delimiter
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.prefetch_buffer_size = prefetch_buffer_size
        self.stride = stride
        self.dataset = tf.data.TextLineDataset(csv_filename).skip(1)
        
    def input_fn(self):
        def split_seq(seq):
            #return dict(zip(["source_seq", "target_seq"],[seq[:self.input_seq_len], seq[self.input_seq_len:]]))
            return seq[:self.input_seq_len], seq[self.input_seq_len:]

        def decode_csv(line):
            parsed_line = tf.decode_csv(line,
                                        record_defaults=COLUMN_DEFAULTS,
                                        field_delim=self.csv_delimiter)

            features = dict(zip(COLUMN_NAMES, parsed_line))
            return tf.stack([features["value"]])

        dataset = self.dataset.map(decode_csv)
        dataset = dataset.apply(tf.contrib.data.sliding_window_batch(
            window_size=self.input_seq_len+self.output_seq_len,
            stride=self.stride))
        dataset = dataset.map(split_seq)
        if self.epoch != None:
            dataset = dataset.repeat(self.epoch)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(buffer_size=self.prefetch_buffer_size)
        iterator = dataset.make_one_shot_iterator()
        features = iterator.get_next()
        return features
