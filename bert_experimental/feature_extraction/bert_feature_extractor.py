import logging
import numpy as np
import tensorflow as tf

from tensorflow.python.estimator.estimator import Estimator
from tensorflow.python.estimator.run_config import RunConfig
from tensorflow.python.estimator.model_fn import EstimatorSpec

from tensorflow.keras.utils import Progbar

from .text_preprocessing import FullTokenizer, convert_lst_to_features, stub_preprocessor

logger = logging.getLogger('tensorflow')
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s :  %(message)s')
sh = logging.StreamHandler()
sh.setLevel(logging.INFO)
sh.setFormatter(formatter)
logger.handlers = [sh]


class BERTFeatureExtractor(object):
    def __init__(self, graph_path, vocab_path,
                 preprocessor=stub_preprocessor, use_gpu=True,
                 batch_size=256, seq_len=64, space_escape='_'):

        self.batch_size = batch_size
        self.seq_len = seq_len
                
        self._tokenizer = FullTokenizer(vocab_path)
        self._preprocessor = preprocessor
        self._graphdef = graph_path

        self._use_gpu = use_gpu
        self._config = self._build_config()
        self._graph = tf.Graph()
        self._sess = tf.Session(graph=self._graph, config=self._config)
        self._input_names = ['input_ids', 'input_mask', 'input_type_ids']
        self._space_escape = space_escape
        self._data_container = DataContainer()
        
        with self._graph.as_default():
            self._estimator = self._build_estimator()
            self._input_fn = self._build_input_fn()
            self._predict_fn = self._estimator.predict(
                input_fn=self._input_fn, yield_single_examples=False)
        self.transform(self._space_escape)
        logger.info('Initialized.')

    def _build_config(self):
        config = tf.ConfigProto(device_count={'GPU': 1 if self._use_gpu else 0})
        config.gpu_options.allow_growth = True
        return config
            
    def _build_estimator(self):
        def model_fn(features, mode):
            with tf.gfile.GFile(self._graphdef, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())

            output = tf.import_graph_def(graph_def,
                                         input_map={k + ':0': features[k] 
                                                    for k in self._input_names},
                                         return_elements=['final_encodes:0'])

            return EstimatorSpec(mode=mode, predictions={'output': output[0]})

        return Estimator(model_fn=model_fn, config=RunConfig(session_config=self._config))

    def _build_input_fn(self):

        def generator():
            while True:
                yield self._build_feed_dict(self._data_container.get())

        def input_fn():
            return tf.data.Dataset.from_generator(
                generator,
                output_types={iname: tf.int32 for iname in self._input_names},
                output_shapes={iname: (None, None) for iname in self._input_names})
        return input_fn
    
    def _build_feed_dict(self, texts):
      
        text_features = list(convert_lst_to_features(
            texts, self.seq_len, self.seq_len, 
            self._tokenizer, is_tokenized=False, mask_cls_sep=False))
        
        target_shape = (len(texts), -1)
        
        feed_dict = {}
        for iname in self._input_names:
            features_i = np.array([getattr(f, iname) for f in text_features])
            features_i = features_i.reshape(target_shape)
            feed_dict[iname] = features_i

        return feed_dict
    
    def transform(self, texts, verbose=False):
      
        if type(texts) is str:
            texts = [texts]

        texts = list(map(self._preprocessor, texts))
        n_samples = len(texts)
        
        blank_idx = []
        for i, text in enumerate(texts):
            if len(text) == 0:
                texts[i] = self._space_escape
                blank_idx.append(i)    

        bar = Progbar(n_samples)
        
        mats = []
        for bi, text_batch in enumerate(batch(texts, self.batch_size)):
          
            self._data_container.set(text_batch)
            features = next(self._predict_fn)['output']
            mats.append(features)
            
            if verbose:
                bar.add(len(text_batch))
    
        mat = np.vstack(mats)
        if len(blank_idx):
            blank_idx = np.array(blank_idx)
            mat[blank_idx] = 0.0

        return mat

    def __call__(self, texts, verbose=False):
        return self.transform(texts, verbose)


class DataContainer:
    def __init__(self):
        self._samples = None

    def set(self, samples):
        self._samples = samples

    def get(self):
        return self._samples


def batch(iterable, n=1):
    itr_len = len(iterable)
    for ndx in range(0, itr_len, n):
        yield iterable[ndx:min(ndx + n, itr_len)]
