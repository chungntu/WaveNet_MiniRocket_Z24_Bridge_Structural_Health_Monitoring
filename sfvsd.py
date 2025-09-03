import tensorflow as tf, time
dev = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
with tf.device(dev):
    a = tf.random.normal([4096,4096]); b = tf.random.normal([4096,4096])
    t0 = time.time(); _ = tf.matmul(a,b).numpy()
print("Using", dev, "| Elapsed(s):", time.time()-t0)
