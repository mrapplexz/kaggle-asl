import json
import subprocess
from pathlib import Path

import pandas as pd
import tensorflow as tf

from kekpipe.model import XYZ_ONE_FRAME
from kekpipe.train import collate_fn, TheDataset

if __name__ == '__main__':
    subprocess.run(['onnx2tf', '-i', 'out.onnx', '-osd', '-coion', '-kat', 'inputs', '-rtpo', 'Erf'])
    with Path('data/sign_to_prediction_index_map.json').open('r') as f:
        sign_map = json.load(f)
    train = pd.read_csv('data/train.csv')
    data_itm = collate_fn([TheDataset(train, sign_map, do_augment=False)[0]])
    mdl = tf.saved_model.load('saved_model')
    c = tf.lite.TFLiteConverter.from_saved_model('saved_model')
    c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    c.optimizations = [tf.lite.Optimize.DEFAULT]
    c.target_spec.supported_types = [tf.float16]
    mdl_tflite = c.convert()
    with open('out.tflite', 'wb') as f:
        f.write(mdl_tflite)
    inte = tf.lite.Interpreter('out.tflite', experimental_preserve_all_tensors=True)
    run = inte.get_signature_runner()
    outs = run(inputs=tf.convert_to_tensor(data_itm['xyz'].reshape(-1, XYZ_ONE_FRAME, 3)))['outputs']
    print(outs)
