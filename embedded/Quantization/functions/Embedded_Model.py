import numpy as np
import time

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    # Full TensorFlow exposes the same Interpreter. Embedded images often
    # install tflite_runtime only; this host has tensorflow instead.
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

class Embedded_Model():
    
    def __init__(self, model_path, select_tf_ops=False, tflite_path=None, num_threads=None):
        '''Basic Init Function, record model path, load the model from model_path and then convert the model to TFLite

        Args:
            model_path (string): Model Path (default TFLite file is ``model_path + '.tflite'``)
            select_tf_ops (bool): Load with the AUTO op resolver when the flatbuffer
                was converted with SELECT_TF_OPS.
            tflite_path (string): Optional flatbuffer path. Use this for the float
                sibling (``model_path + '.float.tflite'``) so the dynamic-range file
                at ``model_path + '.tflite'`` stays in place.
        '''
        self.model_path = model_path
        self.select_tf_ops = select_tf_ops
        self.tflite_path = tflite_path
        self.num_threads = num_threads
        self.interpreter = self.load_tflite_model()
        
    def print_model_details(self):
        '''Print Model details (Layers, resulting data types after each layer operation)
        '''
        for details in self.interpreter.get_tensor_details():                        # Iterates for each model detail
            print(details['name'], details['dtype'])
        print(details.keys())
        
    def set_input_tensor(self, data):
        ''' Load data to be predicted in the model input layer

        Args:
            interpreter (Interpreter): TFLite model
            data ([float]): data to be loaded
        '''
        input_detail = self.interpreter.get_input_details()[0]
        expected_shape = tuple(int(dim) for dim in input_detail['shape'])
        array = np.asarray(data)
        # Batch is 1. Callers may pass either (1, H, W, C) or (H, W, C).
        # The old ``[:, :]`` write dropped the channel axis on rank-4 images.
        if array.shape != expected_shape:
            if expected_shape and expected_shape[0] == 1 and array.shape == expected_shape[1:]:
                array = array.reshape(expected_shape)
            else:
                array = np.reshape(array, expected_shape)
        array = np.ascontiguousarray(array, dtype=input_detail['dtype'])
        self.interpreter.set_tensor(input_detail['index'], array)

    def classify_data(self, data):
        ''' Model classification method

        Args:
            interpreter (Interpreter): TFLite model
            data ([float]): data to be predicted

        Returns:
            output ([int]): Model output list for each data sample

        '''
        self.set_input_tensor(data)                                                  # Load data into Model Input Layer

        self.interpreter.invoke()                                                    # Runs Classification to data loaded on Input Layer
        output_details = self.interpreter.get_output_details()                    # Get model Output Layer expected data format 

        if len(output_details)>1:
            # Get predictions for all output heads
            predictions = []
            for output_detail in output_details:

                output = np.squeeze(self.interpreter.get_tensor(output_detail['index']))    # Format the model output to expected format
                predictions.append(output)
        else:
            output_details = self.interpreter.get_output_details()[0]                    # Get model Output Layer expected data format 
            predictions = np.squeeze(self.interpreter.get_tensor(output_details['index']))    # Format the model output to expected format
        return predictions

    def load_tflite_model(self):
        ''' Load TFLite model

        Returns:
            interpreter (Interpreter): TFLite model
        '''

        tflite_path = self.tflite_path or (self.model_path + '.tflite')
        thread_kwargs = {}
        if self.num_threads:
            thread_kwargs['num_threads'] = int(self.num_threads)
        if self.select_tf_ops:
            import tensorflow as tf
            interpreter = tf.lite.Interpreter(
                model_path=tflite_path,
                experimental_op_resolver_type=tf.lite.experimental.OpResolverType.AUTO,
                **thread_kwargs,
            )
        else:
            interpreter = Interpreter(model_path=tflite_path, **thread_kwargs)
        interpreter.allocate_tensors()                        # Allocate model tensors
        
        return interpreter
    
    def classify_batch_data(self, X_test):
        '''Model classification to data batch

        Args: 
            X_test ([float]): Signal Data Batch

        Returns:
            y_pred ([int]): Prediction array
        '''
        n_classifications = X_test.shape[0]                                 # Get number of classifications

        y_pred = []                                                         # Initialize a empty array of predictions
        t0 = time.time()
        for i in range(n_classifications):                                  # Interates over number of classifications
            print(f'{i+1} out of {n_classifications}', end='\r')            # Debugging
            y_pred.append(self.classify_data(X_test[i]))                    # Appending the prediction for the i-th data sample in Data Batch
        
        y_pred = np.asarray(y_pred)
        # SCE multiclass heads are 0-indexed (SparseCategoricalCrossentropy,
        # gesture remap in Pipeline_deployment_v2). Do not subtract 1.
        # Outputs are logits when the SavedModel was trained with from_logits=True;
        # softmax is monotonic, so argmax is the class id either way.
        if y_pred.ndim < 2:
            raise ValueError(
                'classify_batch_data expected per-sample class scores, got shape %s' % (y_pred.shape,)
            )
        y_pred = np.argmax(y_pred, axis=-1)
        t1=time.time()

        print(f'A classificação de todos as amostras foi de {t1-t0} segundos ou {np.round((t1-t0)/n_classifications, 2)} segundos por amostras')

        return y_pred