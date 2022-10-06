import tensorflow as tf

class BidirectionalLSTM(object):
    def __init__(self, input_dim, output_dim, config):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input((128,)),
            tf.keras.layers.Embedding(input_dim=input_dim, output_dim=output_dim),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
            tf.keras.layers.GlobalMaxPool1D(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(4, activation='softmax')
        ])

        self.model.compile(
            loss=config['train']['criterion'], 
            optimizer=config['train']['optimizer'], 
            metrics=['accuracy']
        )
