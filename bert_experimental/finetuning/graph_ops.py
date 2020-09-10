import tensorflow as tf

from tensorflow.python.framework.graph_util import convert_variables_to_constants
from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference


def load_graph(frozen_graph_filename):
    with tf.io.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)
    return graph
    

### UPD old version to tf2/working with tf 1.x
def freeze_keras_model(model, export_path=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    @param model The Keras model to be optimized for inference.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.compat.v1.graph_util import convert_variables_to_constants
    from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference
    
    session = tf.compat.v1.keras.backend.get_session()
    graph = session.graph
    
    with graph.as_default():

        input_tensors = model.inputs
        output_tensors = model.outputs
        dtypes = [t.dtype.as_datatype_enum for t in input_tensors]
        input_ops = [t.name.rsplit(":", maxsplit=1)[0] for t in input_tensors]
        output_ops = [t.name.rsplit(":", maxsplit=1)[0] for t in output_tensors]
        
        tmp_g = graph.as_graph_def()
        if clear_devices:
            for node in tmp_g.node:
                node.device = ""
        
        tmp_g = optimize_for_inference(
            tmp_g, input_ops, output_ops, dtypes, False)
        
        tmp_g = convert_variables_to_constants(session, tmp_g, output_ops)
        
        if export_path is not None:
            with tf.io.gfile.GFile(export_path, "wb") as f:
                f.write(tmp_g.SerializeToString())
        
        return tmp_g
