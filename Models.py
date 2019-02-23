import tensorflow as tf
from tensorflow.python.keras import backend as K


class Segnet:
    """Implementation of SEGNet, it automatically compile the model,
    if you don't want to see the summary set 'show_summary' to false"""
    def __init__(self, nb_classes, input_shape, show_summary=True):
        self.nb_classes = nb_classes
        self.input_shape = input_shape
        self.model = None
        self.show_summary = show_summary
        self.define_model()

    @staticmethod
    def conv_batch_relu(tensor, filters):
        """Block used multiple time in the model, takes a tensor as input and return another tensor"""
        tensor = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')(tensor)
        tensor = tf.keras.layers.BatchNormalization()(tensor)
        tensor = tf.keras.layers.Activation(activation='relu')(tensor)
        return tensor

    def mean_iou(self, y_true, y_pred):
        """Implementation of mean IoU metric"""
        prec = []
        t = 0.5
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, self.nb_classes)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
        return K.mean(K.stack(prec), axis=0)

    def define_model(self):
        """Define and compile the model"""
        _input_ = tf.keras.layers.Input(self.input_shape, name='input')

        # Encoder
        x = self.conv_batch_relu(_input_, filters=16)
        x = self.conv_batch_relu(x, filters=16)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Dropout(0.5)(x)

        x = self.conv_batch_relu(x, filters=32)
        x = self.conv_batch_relu(x, filters=32)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Dropout(0.5)(x)

        x = self.conv_batch_relu(x, filters=64)
        x = self.conv_batch_relu(x, filters=64)
        x = self.conv_batch_relu(x, filters=64)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Dropout(0.5)(x)

        x = self.conv_batch_relu(x, filters=128)
        x = self.conv_batch_relu(x, filters=128)
        x = self.conv_batch_relu(x, filters=128)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Dropout(0.5)(x)

        x = self.conv_batch_relu(x, filters=256)
        x = self.conv_batch_relu(x, filters=256)
        x = self.conv_batch_relu(x, filters=256)
        x = tf.keras.layers.Dropout(0.5)(x)

        # Decoder
        x = self.conv_batch_relu(x, filters=256)
        x = self.conv_batch_relu(x, filters=256)
        x = self.conv_batch_relu(x, filters=256)
        x = tf.keras.layers.Dropout(0.5)(x)

        x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
        x = self.conv_batch_relu(x, filters=128)
        x = self.conv_batch_relu(x, filters=128)
        x = self.conv_batch_relu(x, filters=128)
        x = tf.keras.layers.Dropout(0.5)(x)

        x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
        x = self.conv_batch_relu(x, filters=64)
        x = self.conv_batch_relu(x, filters=64)
        x = self.conv_batch_relu(x, filters=64)
        x = tf.keras.layers.Dropout(0.5)(x)

        x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
        x = self.conv_batch_relu(x, filters=32)
        x = self.conv_batch_relu(x, filters=32)
        x = tf.keras.layers.Dropout(0.5)(x)

        x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
        x = self.conv_batch_relu(x, filters=16)
        x = self.conv_batch_relu(x, filters=16)
        x = tf.keras.layers.Dropout(0.5)(x)

        x = tf.keras.layers.Conv2D(self.nb_classes, (1, 1), activation='linear', padding='same', name='output')(x)
        x = tf.keras.layers.Reshape((-1, self.nb_classes))(x)
        _output_ = tf.keras.layers.Activation('sigmoid')(x)

        model = tf.keras.models.Model(inputs=_input_, outputs=_output_)
        if self.show_summary:
            print(model.summary())

        model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(lr=1e-3), metrics=[self.mean_iou])
        self.model = model


class Enet:
    """Implementation of ENet, it automatically compile the model.
    If you don't want to see the summary set 'show_summary' to false,
    If you want the faster version of ENet set 'optimized' to True."""
    def __init__(self, nb_classes, input_shape, optimized=False, show_summary=True):
        self.nb_classes = nb_classes
        self.input_shape = input_shape
        self.l1 = 0.0
        self.l2 = 0.0
        self._reduced = optimized
        self._io_filters = 64 if not optimized else 48
        self._filters = 128 if not optimized else 96
        self.model = None
        self.show_summary = show_summary
        self.define_model()

    def initial_block(self, tensor):
        """Initial block after the input layer"""
        conv = tf.keras.layers.Conv2D(filters=13, kernel_size=(3, 3), strides=(2, 2), padding='same',
                                      kernel_regularizer=tf.keras.regularizers.L1L2(self.l1, self.l2))(tensor)
        pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(tensor)
        concat = tf.keras.layers.concatenate([conv, pool], axis=-1)
        return concat

    def bottleneck_encoder(self, tensor, nfilters, dilated=0, downsampling=False,
                           asymmetric=False, normal=False, drate=0.1):
        """Block used multiple times for encoding the image,
        if downsampling is set to true reduces the size of the tensor,
        asymmetric, normal and dilated are parameters used for the convolution, one of the 3 can be selected."""
        y = tensor
        skip = tensor
        stride = 1
        ksize = 1
        if downsampling:
            stride = 2
            ksize = 2
            skip = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(skip)
            skip = tf.keras.layers.Permute((1, 3, 2))(skip)
            ch_pad = nfilters - tf.keras.backend.int_shape(tensor)[-1]
            skip = tf.keras.layers.ZeroPadding2D(padding=((0, 0), (0, ch_pad)))(skip)
            skip = tf.keras.layers.Permute((1, 3, 2))(skip)

        y = tf.keras.layers.Conv2D(filters=nfilters // 4, kernel_size=(ksize, ksize),
                                   strides=(stride, stride), padding='same', use_bias=False,
                                   kernel_regularizer=tf.keras.regularizers.L1L2(self.l1, self.l2))(y)
        y = tf.keras.layers.BatchNormalization(momentum=0.1)(y)
        y = tf.keras.layers.PReLU(shared_axes=[1, 2])(y)

        if normal:
            y = tf.keras.layers.Conv2D(filters=nfilters // 4, kernel_size=(3, 3), padding='same',
                                       kernel_regularizer=tf.keras.regularizers.L1L2(self.l1, self.l2))(y)
        elif asymmetric:
            y = tf.keras.layers.Conv2D(filters=nfilters // 4, kernel_size=(5, 1), padding='same', use_bias=False,
                                       kernel_regularizer=tf.keras.regularizers.L1L2(self.l1, self.l2))(y)
            y = tf.keras.layers.Conv2D(filters=nfilters // 4, kernel_size=(1, 5),  padding='same',
                                       kernel_regularizer=tf.keras.regularizers.L1L2(self.l1, self.l2))(y)
        elif dilated:
            y = tf.keras.layers.Conv2D(filters=nfilters // 4, kernel_size=(3, 3),
                                       dilation_rate=(dilated, dilated), padding='same',
                                       kernel_regularizer=tf.keras.regularizers.L1L2(self.l1, self.l2))(y)
        y = tf.keras.layers.BatchNormalization(momentum=0.1)(y)
        y = tf.keras.layers.PReLU(shared_axes=[1, 2])(y)

        y = tf.keras.layers.Conv2D(filters=nfilters, kernel_size=(1, 1), use_bias=False,
                                   kernel_regularizer=tf.keras.regularizers.L1L2(self.l1, self.l2))(y)
        y = tf.keras.layers.BatchNormalization(momentum=0.1)(y)
        y = tf.keras.layers.SpatialDropout2D(rate=drate)(y)

        y = tf.keras.layers.Add()([y, skip])
        y = tf.keras.layers.PReLU(shared_axes=[1, 2])(y)

        return y

    def bottleneck_decoder(self, tensor, nfilters, upsampling=False, normal=False):
        """Block used multiple times for encoding the image,
        if upsampling is set to true it increase the size of the tensor,
        if normal is set to true it adds a convolution layer.
        The choices are upsampling, normal or none of them."""
        y = tensor
        skip = tensor
        if upsampling:
            skip = tf.keras.layers.Conv2D(filters=nfilters, kernel_size=(1, 1),
                                          strides=(1, 1), padding='same', use_bias=False,
                                          kernel_regularizer=tf.keras.regularizers.L1L2(self.l1, self.l2))(skip)
            skip = tf.keras.layers.UpSampling2D(size=(2, 2))(skip)

        y = tf.keras.layers.Conv2D(filters=nfilters // 4, kernel_size=(1, 1),
                                   strides=(1, 1), padding='same', use_bias=False,
                                   kernel_regularizer=tf.keras.regularizers.L1L2(self.l1, self.l2))(y)
        y = tf.keras.layers.BatchNormalization(momentum=0.1)(y)
        y = tf.keras.layers.PReLU(shared_axes=[1, 2])(y)

        if upsampling:
            y = tf.keras.layers.Conv2DTranspose(filters=nfilters // 4, kernel_size=(3, 3),
                                                strides=(2, 2), padding='same',
                                                kernel_regularizer=tf.keras.regularizers.L1L2(self.l1, self.l2))(y)
        elif normal:
            tf.keras.layers.Conv2D(filters=nfilters // 4, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                   kernel_regularizer=tf.keras.regularizers.L1L2(self.l1, self.l2))(y)
        y = tf.keras.layers.BatchNormalization(momentum=0.1)(y)
        y = tf.keras.layers.PReLU(shared_axes=[1, 2])(y)

        y = tf.keras.layers.Conv2D(filters=nfilters, kernel_size=(1, 1), use_bias=False,
                                   kernel_regularizer=tf.keras.regularizers.L1L2(self.l1, self.l2))(y)
        y = tf.keras.layers.BatchNormalization(momentum=0.1)(y)

        y = tf.keras.layers.Add()([y, skip])
        y = tf.keras.layers.ReLU()(y)

        return y

    def mean_iou(self, y_true, y_pred):
        """Implementation of mean IoU metric"""
        prec = []
        t = 0.5
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, self.nb_classes)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
        return K.mean(K.stack(prec), axis=0)

    def define_model(self):
        """Define and compile the model"""
        _input_ = tf.keras.layers.Input(self.input_shape)

        if self._reduced:
            x = tf.keras.layers.MaxPooling2D(pool_size=(4, 4), padding='same')(_input_)
            x = self.initial_block(x)
        else:
            x = self.initial_block(_input_)

        x = self.bottleneck_encoder(x, self._io_filters, downsampling=True, normal=True, drate=0.01)
        for _ in range(1, 5):
            x = self.bottleneck_encoder(x, self._io_filters, normal=True, drate=0.01)

        x = self.bottleneck_encoder(x, self._filters, downsampling=True, normal=True)
        x = self.bottleneck_encoder(x, self._filters, normal=True)
        x = self.bottleneck_encoder(x, self._filters, dilated=2)
        x = self.bottleneck_encoder(x, self._filters, asymmetric=True)
        x = self.bottleneck_encoder(x, self._filters, dilated=4)
        x = self.bottleneck_encoder(x, self._filters, normal=True)
        x = self.bottleneck_encoder(x, self._filters, dilated=8)
        x = self.bottleneck_encoder(x, self._filters, asymmetric=True)
        x = self.bottleneck_encoder(x, self._filters, dilated=16)

        x = self.bottleneck_encoder(x, self._filters, normal=True)
        x = self.bottleneck_encoder(x, self._filters, dilated=2)
        x = self.bottleneck_encoder(x, self._filters, asymmetric=True)
        x = self.bottleneck_encoder(x, self._filters, dilated=4)
        x = self.bottleneck_encoder(x, self._filters, normal=True)
        x = self.bottleneck_encoder(x, self._filters, dilated=8)
        x = self.bottleneck_encoder(x, self._filters, asymmetric=True)
        x = self.bottleneck_encoder(x, self._filters, dilated=16)

        x = self.bottleneck_decoder(x, self._io_filters, upsampling=True)
        x = self.bottleneck_decoder(x, self._io_filters, normal=True)
        x = self.bottleneck_decoder(x, self._io_filters, normal=True)

        x = self.bottleneck_decoder(x, 16, upsampling=True)
        x = self.bottleneck_decoder(x, 16, normal=True)

        x = tf.keras.layers.Conv2DTranspose(self.nb_classes, kernel_size=(2, 2), strides=(2, 2), padding='same')(x)
        if self._reduced:
            x = tf.keras.layers.UpSampling2D(size=(4, 4))(x)

        _output_ = tf.keras.layers.Reshape((-1, self.nb_classes))(x)
        _output_ = tf.keras.layers.Activation('sigmoid')(_output_)

        model = tf.keras.models.Model(inputs=_input_, outputs=_output_)
        if self.show_summary:
            print(model.summary())

        model.compile(loss="binary_crossentropy", metrics=[self.mean_iou],
                      optimizer=tf.keras.optimizers.Adam(lr=1e-3))
        self.model = model
