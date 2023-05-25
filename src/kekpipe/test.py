import time

import numpy as np
import pandas as pd

if __name__ == '__main__':
    ROWS_PER_FRAME = 543  # number of landmarks per frame


    def load_relevant_data_subset(pq_path):
        data_columns = ['x', 'y', 'z']
        data = pd.read_parquet(pq_path, columns=data_columns)
        n_frames = int(len(data) / ROWS_PER_FRAME)
        data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
        return data.astype(np.float32)

    data = load_relevant_data_subset('data/train_landmark_files/2044/2426942750.parquet')

    import tflite_runtime.interpreter as tflite

    interpreter = tflite.Interpreter('out.tflite')

    found_signatures = list(interpreter.get_signature_list().keys())

    if 'serving_default' not in found_signatures:
        raise ValueError('Required input signature not found.')

    prediction_fn = interpreter.get_signature_runner("serving_default")
    t1 = time.time()
    output = prediction_fn(inputs=data)
    t2 = time.time()
    print((t2 - t1) * 1000, 'ms')
    sign = np.argmax(output["outputs"])

    print(sign)
