import os
import tensorflow as tf
import tensorflow_hub as hub

from .text_preprocessing import build_preprocessor


class BertLayer(tf.keras.layers.Layer):
    def __init__(self, bert_path, seq_len=64, n_tune_layers=3, 
                 pooling="cls", do_preprocessing=True, verbose=False,
                 tune_embeddings=False, trainable=True, use_layers=None, 
                 as_dict=False, **kwargs):

        self.trainable = trainable
        self.n_tune_layers = n_tune_layers
        self.tune_embeddings = tune_embeddings
        self.do_preprocessing = do_preprocessing

        self.as_dict = as_dict
        self.verbose = verbose
        self.seq_len = seq_len
        self.pooling = pooling
        self.bert_path = bert_path
        self.use_layers = use_layers

        self.var_per_encoder = 16
        if self.pooling not in ["cls", "mean", "sqrt_mean", None]:
            raise NameError(
                f"Undefined pooling type (must be either 'cls', 'mean', 'sqrt_mean' or None, but is {self.pooling}"
            )

        super(BertLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        self.bert = hub.Module(self.build_abspath(self.bert_path), 
                               trainable=self.trainable, name=f"{self.name}_module")

        trainable_layers = []
        if self.tune_embeddings:
            trainable_layers.append("embeddings")

        if self.pooling == "cls":
            trainable_layers.append("pooler")

        if self.n_tune_layers > 0:
            encoder_var_names = [var.name for var in self.bert.variables if 'encoder' in var.name]
            n_encoder_layers = int(len(encoder_var_names) / self.var_per_encoder)
            if self.use_layers:
                n_encoder_layers = min(self.use_layers, n_encoder_layers)
            for i in range(self.n_tune_layers):
                trainable_layers.append(f"encoder/layer_{str(n_encoder_layers - 1 - i)}/")

        # Add module variables to layer's trainable weights
        for var in self.bert.variables:
            if any([l in var.name for l in trainable_layers]):
                self._trainable_weights.append(var)
            else:
                self._non_trainable_weights.append(var)

        if self.verbose:
            print("*** TRAINABLE VARS *** ")
            for var in self._trainable_weights:
                print(var)

        self.build_preprocessor()
        self.initialize_module()

        super(BertLayer, self).build(input_shape)

    def build_abspath(self, path):
        if path.startswith("https://") or path.startswith("gs://"):
            return path
        else:
            return os.path.abspath(path)

    def build_preprocessor(self):
        sess = tf.compat.v1.keras.backend.get_session()
        tokenization_info = self.bert(signature="tokenization_info", as_dict=True)
        vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                              tokenization_info["do_lower_case"]])
        self.preprocessor = build_preprocessor(vocab_file, self.seq_len, do_lower_case)

    def initialize_module(self):
        sess = tf.compat.v1.keras.backend.get_session()

        vars_initialized = sess.run([tf.compat.v1.is_variable_initialized(var) 
                                     for var in self.bert.variables])

        uninitialized = []
        for var, is_initialized in zip(self.bert.variables, vars_initialized):
            if not is_initialized:
                uninitialized.append(var)

        if len(uninitialized):
            sess.run(tf.compat.v1.variables_initializer(uninitialized))

    def call(self, input):

        if self.do_preprocessing:
            input = tf.numpy_function(self.preprocessor, 
                                      [input], [tf.int32, tf.int32, tf.int32], 
                                      name='preprocessor')
            for feature in input:
                feature.set_shape((None, self.seq_len))

        input_ids, input_mask, segment_ids = input

        bert_inputs = dict(
            input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids
        )
        output = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)
        
        input_mask = tf.cast(input_mask, tf.float32)
            
        seq_output = output["sequence_output"]
        tok_output = mul_mask(output.get("token_output", seq_output), input_mask)
        
        if self.pooling == "cls":
            pooled = output["pooled_output"]
        else:
            if self.pooling == "mean":
                pooled = masked_reduce_mean(seq_output, input_mask)

            elif self.pooling == "sqrt_mean":
                pooled = masked_reduce_sqrt_mean(seq_output, input_mask)

            else:
                pooled = mul_mask(seq_output, input_mask)

        if self.as_dict:
            output = {
                "sequence_output": seq_output,
                "pooled_output": pooled,
                "token_output": tok_output
            }
        else:
            output = pooled

        return output

    def get_config(self):
        config_dict = {
            "bert_path": self.bert_path, 
            "seq_len": self.seq_len,
            "pooling": self.pooling,
            "n_tune_layers": self.n_tune_layers,
            "tune_embeddings": self.tune_embeddings,
            "do_preprocessing": self.do_preprocessing,
            "use_layers": self.use_layers,
            "trainable": self.trainable,
            "as_dict": self.as_dict,
            "verbose": self.verbose
        }
        super(BertLayer, self).get_config()
        return config_dict


class StatefulBertLayer(tf.keras.layers.Layer):
    def __init__(self, bert_path, seq_len=64, n_tune_layers=3, 
                 pooling="cls", do_preprocessing=True, verbose=False,
                 tune_embeddings=False, trainable=True, use_layers=None, 
                 as_dict=False, **kwargs):

        self.trainable = trainable
        self.n_tune_layers = n_tune_layers
        self.tune_embeddings = tune_embeddings
        self.do_preprocessing = do_preprocessing

        self.as_dict = as_dict
        self.verbose = verbose
        self.seq_len = seq_len
        self.pooling = pooling
        self.bert_path = bert_path
        self.use_layers = use_layers

        self.var_per_encoder = 16
        if self.pooling not in ["cls", "mean", "sqrt_mean", None]:
            raise NameError(
                f"Undefined pooling type (must be either 'cls', 'mean', 'sqrt_mean' or None, but is {self.pooling}"
            )

        super(StatefulBertLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        self.bert = hub.Module(self.build_abspath(self.bert_path), 
                               trainable=self.trainable, name=f"{self.name}_module")

        trainable_layers = []
        if self.tune_embeddings:
            trainable_layers.append("embeddings")

        if self.pooling == "cls":
            trainable_layers.append("pooler")

        if self.n_tune_layers > 0:
            encoder_var_names = [var.name for var in self.bert.variables if 'encoder' in var.name]
            n_encoder_layers = int(len(encoder_var_names) / self.var_per_encoder)
            if self.use_layers:
                n_encoder_layers = min(self.use_layers, n_encoder_layers)
            for i in range(self.n_tune_layers):
                trainable_layers.append(f"encoder/layer_{str(n_encoder_layers - 1 - i)}/")

        # Add module variables to layer's trainable weights
        for var in self.bert.variables:
            if any([l in var.name for l in trainable_layers]):
                self._trainable_weights.append(var)
            else:
                self._non_trainable_weights.append(var)

        if self.verbose:
            print("*** TRAINABLE VARS *** ")
            for var in self._trainable_weights:
                print(var)

        self.build_preprocessor()
        self.initialize_module()

        super(StatefulBertLayer, self).build(input_shape)

    def build_abspath(self, path):
        if path.startswith("https://") or path.startswith("gs://"):
            return path
        else:
            return os.path.abspath(path)

    def build_preprocessor(self):
        sess = tf.compat.v1.keras.backend.get_session()
        tokenization_info = self.bert(signature="tokenization_info", as_dict=True)
        vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                              tokenization_info["do_lower_case"]])
        self.preprocessor = build_preprocessor(vocab_file, self.seq_len, do_lower_case)

    def initialize_module(self):
        sess = tf.compat.v1.keras.backend.get_session()

        vars_initialized = sess.run([tf.compat.v1.is_variable_initialized(var) 
                                     for var in self.bert.variables])

        uninitialized = []
        for var, is_initialized in zip(self.bert.variables, vars_initialized):
            if not is_initialized:
                uninitialized.append(var)

        if len(uninitialized):
            sess.run(tf.compat.v1.variables_initializer(uninitialized))

    def call(self, input):

        if self.do_preprocessing:
            input_text, input_state = input
            
            preprocessed_text = tf.numpy_function(
                self.preprocessor, [input_text], 
                [tf.int32, tf.int32, tf.int32], 
                name='preprocessor')
            for feature in preprocessed_text:
                feature.set_shape((None, self.seq_len))
            input_ids, input_mask, segment_ids = preprocessed_text
                
        else:
            input_ids, input_mask, segment_ids, input_state = input
            
        bert_inputs = dict(
            input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, input_state=input_state
        )
        output = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)
        
        input_mask = tf.cast(input_mask, tf.float32)
            
        seq_output = output["sequence_output"]
        tok_output = mul_mask(output.get("token_output", seq_output), input_mask)
        
        if self.pooling == "cls":
            pooled = output["pooled_output"]
        else:
            if self.pooling == "mean":
                pooled = masked_reduce_mean(seq_output, input_mask)

            elif self.pooling == "sqrt_mean":
                pooled = masked_reduce_sqrt_mean(seq_output, input_mask)

            else:
                pooled = mul_mask(seq_output, input_mask)

        if self.as_dict:
            output["pooled_output"] = pooled
        else:
            output = pooled

        return output

    def get_config(self):
        config_dict = {
            "bert_path": self.bert_path, 
            "seq_len": self.seq_len,
            "pooling": self.pooling,
            "n_tune_layers": self.n_tune_layers,
            "tune_embeddings": self.tune_embeddings,
            "do_preprocessing": self.do_preprocessing,
            "use_layers": self.use_layers,
            "trainable": self.trainable,
            "as_dict": self.as_dict,
            "verbose": self.verbose
        }
        super(StatefulBertLayer, self).get_config()
        return config_dict
    
def mul_mask(x, m):
    return x * tf.expand_dims(m, axis=-1)

def masked_reduce_mean(x, m):
    return tf.reduce_sum(mul_mask(x, m), axis=1) / (
        tf.reduce_sum(m, axis=1, keepdims=True) + 1e-10)
    
def masked_reduce_sqrt_mean(x, m):
    return tf.reduce_sum(mul_mask(x, m), axis=1) / (
        tf.sqrt(tf.reduce_sum(m, axis=1, keepdims=True)) + 1e-10)


