import tensorflow as tf

class Seq2SeqModel:
    def __init__(self, input_dim, output_dim,
                 input_seq_len, output_seq_len,
                 hidden_size, layer_cnt):
        self.input_dim = input_dim
        self.input_seq_len = input_seq_len
        self.output_dim = output_dim
        self.output_seq_len = output_seq_len
        self.hidden_size = hidden_size
        self.layer_cnt = layer_cnt

    def _add_placeholders(self, features, labels, mode):
        #features = tf.Print(features, [features], "features=")
        #labels = tf.Print(labels, [labels], "labels=")
        self.batch_size = tf.shape(features)[0]
        encoder_inputs = features # [batch_size, seq_len, input_dim]
        self.encoder_inputs = tf.transpose(encoder_inputs, [1,0,2]) # [seq_len, batch_size, input_dim]
        #self.encoder_inputs = tf.Print(self.encoder_inputs, [self.encoder_inputs], "enconder_input=")
        if mode != tf.estimator.ModeKeys.PREDICT:
            self.decoder_targets = tf.transpose(labels, [1,0,2])
            self.decoder_inputs = tf.concat(
                [tf.expand_dims(tf.zeros_like(self.decoder_targets[0]), 0),
                 self.decoder_targets[:-1,:,:]], axis=0)
            #self.decoder_inputs = tf.Print(self.decoder_inputs,[self.decoder_inputs], "decoder_inputs=")

    def build_graph(self, features, labels, params, mode):

        self._add_placeholders(features, labels, mode)

        with tf.variable_scope('encoder') as scope:
            source_seq_len = tf.fill([self.batch_size], self.input_seq_len)
            encoder_cell = tf.contrib.rnn.LSTMCell(
                self.hidden_size)
            encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
                cell=encoder_cell,
                inputs=self.encoder_inputs,
                dtype=tf.float32,
                sequence_length=source_seq_len,
                time_major=True)

        with tf.variable_scope('decoder') as scope:
            decoder_cell = tf.contrib.rnn.LSTMCell(self.hidden_size)
            target_seq_len = tf.fill([self.batch_size], self.output_seq_len)
            if mode == tf.estimator.ModeKeys.PREDICT:
                def initialize_fn():
                    finished = tf.tile([False], [self.batch_size])
                    start_inputs = tf.fill([self.batch_size, self.output_dim], 0.)
                    return (finished, start_inputs)

                def sample_fn(time, outputs, state):
                    return tf.constant([0])

                def next_inputs_fn(time, outputs, state, sample_ids):
                    finished = time >= target_seq_len
                    next_inputs = outputs
                    return (finished, next_inputs, state)

                helper = tf.contrib.seq2seq.CustomHelper(
                    initialize_fn = initialize_fn,
                    sample_fn = sample_fn,                      
                    next_inputs_fn = next_inputs_fn)

                output_layer = tf.layers.Dense(self.output_dim)

                decoder = tf.contrib.seq2seq.BasicDecoder(
                    decoder_cell, helper, encoder_final_state, output_layer)

                decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
                    decoder, output_time_major=True)
                
                self.predictions = decoder_outputs.rnn

            else:
                helper = tf.contrib.seq2seq.TrainingHelper(inputs=self.decoder_inputs,
                                                           sequence_length=target_seq_len,
                                                           time_major=True)

                output_layer = tf.layers.Dense(self.output_dim)

                decoder = tf.contrib.seq2seq.BasicDecoder(
                    decoder_cell, helper, encoder_final_state, output_layer)
                decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
                    decoder, output_time_major=True)

                self.predictions = decoder_outputs.rnn_output

                with tf.variable_scope('loss') as scope:
                    self.loss = tf.losses.mean_squared_error(self.predictions, self.decoder_targets)
                    l2 = params['lambda_l2_reg'] * sum(
                        tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables() if not ("noreg" in tf_var.name or
                                                                                             "Bias" in tf_var.name))
                    self.loss += l2

    def create_model_fn(self):
        def model_fn(features, labels, params, mode):
            # build the seq2seq graph for time-series
            self.build_graph(features, labels, params, mode)

            if mode == tf.estimator.ModeKeys.PREDICT:
                return None

            eval_metric_ops = {
                "rmse": tf.metrics.mean_squared_error(
                    tf.cast(self.decoder_targets, tf.float32), self.predictions)
            }

            #if mode == tf.estimator.ModeKeys.TRAIN:
                
            train_op = tf.contrib.layers.optimize_loss(
                loss=self.loss,
                global_step=tf.contrib.framework.get_global_step(),
                learning_rate=params['learning_rate'], 
                clip_gradients=params['gradient_clipping'],
                optimizer=params['optimizer'])
                
            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=self.loss,
                train_op=train_op,
                eval_metric_ops=eval_metric_ops)

        return model_fn
            
