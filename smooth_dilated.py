from keras import backend as K
from keras.layers import Layer

class SmoothedDilatedLayer1D(Layer):
    """
    1D-convolution for sequence data
    """

    def __init__(self, kernel_size, output_dim, dilation_factor, biased=False, **kwargs):
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.dilation_factor = dilation_factor
        self.biased = biased
        super(SmoothedDilatedLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        c = input_shape[-1]
        
        self.w = self.add_weight(shape=(self.kernel_size, c, self.output_dim), initializer="uniform", name='w')
        self.fix_w = self.add_weight(shape=(self.dilation_factor, self.dilation_factor), initializer="ones", name='fix_w')
        if self.biased:
            self.b = self.add_weight(shape=(self.output_dim, ), initializer="uniform", name='biases')
        self.built = True

    def call(self, x):   
        L = K.shape(x)[1]
        pad_right = (self.dilation_factor - L % self.dilation_factor) if L % self.dilation_factor != 0 else 0

        pad = [[0, pad_right]]
        # decomposition to smaller-sized feature maps
        #[N,L,C] -> [N*d, L/d, C]
        o = K.tf.space_to_batch_nd(x, paddings=pad, block_shape=[self.dilation_factor])
            
        s = 1
        o = K.conv1d(o, self.w, s, padding='same')
		
        l = K.tf.split(o, self.dilation_factor, axis=0)
        res = []
        for i in range(0, self.dilation_factor):
            res.append(self.fix_w[0, i] * l[i])
            for j in range(1, self.dilation_factor):
                res[i] += self.fix_w[j, i] * l[j]	        
        o = K.tf.concat(res, axis=0)
        if biased:
            o = bias_add(o, b)
        o = K.tf.batch_to_space_nd(o, crops=pad, block_shape=[self.dilation_factor])
        
        return o

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)
    
    
# Test Cells
# Just test the smoothedDilatedLayer function
if __name__ == "__main__":
    import keras
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation, Reshape
    from keras.optimizers import SGD
    
    # 生成虚拟数据
    import numpy as np
    x_train = np.random.random((1000, 20, 20))
    print(x_train.shape)
    y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
    print(y_train.shape)
    x_test = np.random.random((100, 20, 20))
    y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)
    
    model = Sequential()
    # Dense(64) 是一个具有 64 个隐藏神经元的全连接层。
    # 在第一层必须指定所期望的输入数据尺寸：
    # 在这里，是一个 20 维的向量。
    model.add(SmoothedDilatedLayer(kernel_size=3, output_dim=64, dilation_factor=2, name="smoothed_dilated_l1"))
    model.add(Reshape((-1,)))
    model.add(Dense(64, activation='relu', input_shape=(64, )))
    model.add(Dense(10, activation='softmax', input_dim=64))
    
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    
    model.fit(x_train, y_train,
              epochs=100,
              batch_size=128)
    score = model.evaluate(x_test, y_test, batch_size=128)
    print(model.summary())
    print(score)