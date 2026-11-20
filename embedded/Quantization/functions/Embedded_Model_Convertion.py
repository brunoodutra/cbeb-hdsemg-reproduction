import os
import tensorflow as tf
import time
from Embedded_Model import Embedded_Model


def convert_saved_model_to_tflite(model_path, output_path=None):
    '''Dynamic-range weight quantization via ``tf.lite.Optimize.DEFAULT``.

    Same conversion path as ``Embedded_Model_Convertion``: 
    ``TFLiteConverter.from_saved_model`` plus default optimizations
    (int8 weights, float activations and I/O). This is not full-integer
    INT8. ``SELECT_TF_OPS`` is enabled only if the builtin conversion fails.

    The flatbuffer is written to ``{model_path}.tflite`` (sibling of a
    SavedModel directory), matching the historical converter.

    Returns:
        dict: tflite_path, quant_mode, select_tf_ops, num_bytes
    '''
    model_path = os.path.abspath(os.path.normpath(model_path))
    if output_path is None:
        output_path = model_path + '.tflite'

    def _convert(select_tf_ops):
        converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if select_tf_ops:
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS,
            ]
            converter.allow_custom_ops = True
        return converter.convert()

    select_tf_ops = False
    quant_mode = 'dynamic_range_optimize_default'
    try:
        tflite_model = _convert(False)
    except Exception as builtin_error:
        print('Builtin TFLite conversion failed: %s' % builtin_error)
        print('Retrying with SELECT_TF_OPS.')
        try:
            tflite_model = _convert(True)
        except Exception as select_error:
            raise RuntimeError(
                'TFLite conversion failed for %s.\n'
                'Builtin error: %s\n'
                'SELECT_TF_OPS error: %s' % (model_path, builtin_error, select_error)
            ) from select_error
        select_tf_ops = True
        quant_mode = 'dynamic_range_optimize_default_select_tf_ops'

    with open(output_path, 'wb') as handle:
        handle.write(tflite_model)
    return {
        'tflite_path': output_path,
        'quant_mode': quant_mode,
        'select_tf_ops': select_tf_ops,
        'num_bytes': len(tflite_model),
    }


def convert_saved_model_to_float_tflite(model_path, output_path=None):
    '''Float32 TFLite with no converter optimizations.

    ``TFLiteConverter.from_saved_model`` is left at its default: no
    ``optimizations``, no representative dataset, no integer quantization.
    Weights and activations stay float32. The flatbuffer is written to
    ``{model_path}.float.tflite`` so it does not replace the dynamic-range
    sibling ``{model_path}.tflite``.

    Returns:
        dict: tflite_path, quant_mode, select_tf_ops, num_bytes
    '''
    model_path = os.path.abspath(os.path.normpath(model_path))
    if output_path is None:
        output_path = model_path + '.float.tflite'

    def _convert(select_tf_ops):
        converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
        # Do not set converter.optimizations. DEFAULT would dynamic-range
        # quantize the weights and confound the latency comparison.
        if select_tf_ops:
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS,
            ]
            converter.allow_custom_ops = True
        return converter.convert()

    select_tf_ops = False
    quant_mode = 'float32_no_optimize'
    try:
        tflite_model = _convert(False)
    except Exception as builtin_error:
        print('Builtin float TFLite conversion failed: %s' % builtin_error)
        print('Retrying with SELECT_TF_OPS.')
        try:
            tflite_model = _convert(True)
        except Exception as select_error:
            raise RuntimeError(
                'Float TFLite conversion failed for %s.\n'
                'Builtin error: %s\n'
                'SELECT_TF_OPS error: %s' % (model_path, builtin_error, select_error)
            ) from select_error
        select_tf_ops = True
        quant_mode = 'float32_no_optimize_select_tf_ops'

    with open(output_path, 'wb') as handle:
        handle.write(tflite_model)
    return {
        'tflite_path': output_path,
        'quant_mode': quant_mode,
        'select_tf_ops': select_tf_ops,
        'num_bytes': len(tflite_model),
    }


class Embedded_Model_Convertion(Embedded_Model):

    def __init__(self, model_path):
        '''Basic Init Function, record model path, load the model from model_path and then convert the model to TFLite

        Args:
            model_path (string): Model Path
        '''
        self.compatibility_detection()
        self.model_path = model_path
        self.convert_model_tflite()
        self.interpreter = self.load_tflite_model()
        
    def compatibility_detection(self):
        ''' Verify the compatibility of the tf and TFLite versions
        '''  
        
        @tf.lite.experimental.authoring.compatible
        @tf.function(input_signature=[
            tf.TensorSpec(shape=[None], dtype=tf.float32)
        ])
        def f(x):
            return tf.cosh(x)

        # Evaluate the tf.function
        result = f(tf.constant([0.0]))
        print (f"result = {result}")
        
    def convert_model_tflite(self):
        ''' Convert and Optimize the Keras model to TFLite model
        '''
        t0 = time.time()
        converted = convert_saved_model_to_tflite(self.model_path)
        self.tflite_output_path = converted['tflite_path']
        self.quant_mode = converted['quant_mode']
        self.select_tf_ops = converted['select_tf_ops']
        t1 = time.time()
        print(
            'A conversão do modelo durou %s segundos. mode=%s path=%s'
            % (t1 - t0, self.quant_mode, self.tflite_output_path)
        )