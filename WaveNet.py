import tensorflow as tf
import os
from tensorflow.keras import layers 
from tensorflow.keras.layers import Flatten

# ============================================================
# WaveNetResidual(num_filters, kernel_size, dilation_rate)
# ------------------------------------------------------------
# Hàm xây dựng 1 residual block theo phong cách WaveNet:
#   - Gồm 2 Conv1D song song: activation tanh và sigmoid
#   - Nhân chập 2 nhánh (Multiply) -> gating mechanism
#   - Conv1D kernel=1 để điều chỉnh kênh
#   - Add với input ban đầu để tạo residual connection
#   - Trả về: (residual output, skip connection)
# ============================================================
def WaveNetResidual(num_filters, kernel_size, dilation_rate):        
    """A WaveNet-like residual block."""
    def build_residual_block(x_init):
        # Nhánh tanh
        tanh_out = layers.Conv1D(
            num_filters, kernel_size, dilation_rate=dilation_rate, 
            padding='causal', activation='tanh'
        )(x_init)
        # Nhánh sigmoid
        sigm_out = layers.Conv1D(
            num_filters, kernel_size, dilation_rate=dilation_rate, 
            padding='causal', activation='sigmoid'
        )(x_init)
        # Kết hợp gating (element-wise multiply)
        x = layers.Multiply()([tanh_out, sigm_out])
        # Conv1D kernel=1 để tái tạo đặc trưng
        x = layers.Conv1D(num_filters, 1, padding='causal')(x)
        # Residual: cộng với input gốc
        res_x = layers.Add()([x, x_init])
        return res_x, x  # residual output và skip connection
    return build_residual_block

# ============================================================
# build_wavenet_model_residual_blocks(...)
# ------------------------------------------------------------
# Xây dựng mô hình WaveNet nhiều residual blocks
# Input:
#   - input_shape: dạng (width,1)
#   - num_classes: số lớp output
#   - num_filters: số filters cho Conv1D
#   - numberOfBlocks: số block (mỗi block có nhiều residuals)
#   - numberOfResidualsPerBlock: số residual mỗi block (với dilation 2^k)
#   - kernel_size: kích thước kernel cho Conv1D
# Output:
#   - model Keras hoàn chỉnh
# ============================================================
def build_wavenet_model_residual_blocks(input_shape, num_classes, num_filters, numberOfBlocks, numberOfResidualsPerBlock, kernel_size=2):
    inputs = layers.Input(shape=input_shape)

    # Lớp conv đầu tiên (causal)
    x = layers.Conv1D(
        filters=num_filters, kernel_size=kernel_size, 
        dilation_rate=1, padding='causal', activation='relu', 
        input_shape=input_shape
    )(inputs)

    totalLayers = numberOfBlocks * numberOfResidualsPerBlock  
    # Tạo các residual block liên tiếp
    for i in range(totalLayers):
        k = i % numberOfResidualsPerBlock  
        # VD: với 3 block và 10 residuals: sẽ có 2^0..2^9, lặp lại 3 lần
        x, skip = WaveNetResidual(num_filters, kernel_size, 2**k)(x)
        if i == 0:
            skips = skip
        else:
            skips = layers.Add()([skips, skip])

    # Classification head
    x = layers.Activation('relu')(skips)
    x = layers.Conv1D(num_filters, 1, activation='relu')(x)
    x = layers.Conv1D(num_filters, 1, activation='relu')(x)
    x = layers.Dense(num_filters, activation='relu')(x)    
    x = Flatten()(x)   
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# ============================================================
# WavenetRun(...)
# ------------------------------------------------------------
# Chạy huấn luyện mô hình WaveNet
# Input:
#   - model: nếu None thì build mới, nếu không thì train tiếp
#   - filters, batch_size, epochs, learning_rate, ...
#   - width: chiều dài input (65536)
#   - classes: danh sách nhãn
#   - X_train, X_val, X_test, y_train, y_val, y_test: dữ liệu
# Output:
#   - Huấn luyện, lưu checkpoint (.h5), lưu history (.txt)
#   - Tên file lưu rõ ràng: chứa filters, blocks, res, width, classes, epochs, batch_size
# ============================================================
def WavenetRun(model, filters, batch_size, epochs, learning_rate, numberOfResidualsPerBlock, numberOfBlocks, width, classes, X_train, X_val,  X_test, y_train, y_val, y_test):  
    # Định nghĩa input shape và số lớp output
    input_shape = (width, 1)  
    num_classes = len(classes) 
    
    # Nếu model chưa có -> build mới
    if model is None:
        model = build_wavenet_model_residual_blocks(
            input_shape, num_classes, filters, 
            numberOfBlocks, numberOfResidualsPerBlock
        )    
    
    from tensorflow.keras.optimizers import Adam

    # Optimizer Adam với learning_rate custom
    custom_learning_rate = learning_rate
    custom_adam_optimizer = Adam(learning_rate=custom_learning_rate)

    model.compile(
        optimizer=custom_adam_optimizer, 
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )

    # In summary mô hình
    model.summary()

    # --------------------------------------------------------
    # Sinh tên file mô hình đầy đủ và rõ nghĩa
    # --------------------------------------------------------
    def get_checkpoint_filename(base_name, version=1):
        """
        Sinh tên file .h5, nếu trùng thì tăng version
        """
        while True:
            checkpoint_name = f"{base_name}_v{version}.h5"
            if not os.path.exists(checkpoint_name):                
                return checkpoint_name
            version += 1

    # Tạo base_name với đầy đủ thông tin siêu tham số
    base_name = (
        f"WaveNet_filters{filters}"
        f"_blocks{numberOfBlocks}"
        f"_res{numberOfResidualsPerBlock}"
        f"_width{width}"
        f"_classes{num_classes}"
        f"_ep{epochs}"
        f"_bs{batch_size}"
    )

    # Lấy tên file cuối cùng (có thêm version nếu trùng)
    final_checkpoint_name = get_checkpoint_filename(base_name)
    base_filename = final_checkpoint_name  # dùng cho cả history và model

    # --------------------------------------------------------
    # Callbacks: EarlyStopping + ModelCheckpoint
    # --------------------------------------------------------
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=150, verbose=1, mode='min'
        ),
        tf.keras.callbacks.ModelCheckpoint(
            final_checkpoint_name, monitor='val_loss', 
            save_best_only=True, mode='min', verbose=1
        )
    ]

    # --------------------------------------------------------
    # Huấn luyện mô hình
    # --------------------------------------------------------
    history = model.fit(
        X_train, y_train, 
        epochs=epochs, batch_size=batch_size, 
        validation_data=(X_val, y_val),
        callbacks=callbacks, verbose=1
    )  
    
    # --------------------------------------------------------
    # Lưu lại history với tên file khớp model (chỉ khác đuôi .txt)
    # --------------------------------------------------------
    history_filename = f'./History/{os.path.basename(base_filename).replace(".h5", ".txt")}'
    with open(history_filename, "w") as file:
        file.write(str(history.history))
    print(f"[INFO] Saved training history to {history_filename}")
