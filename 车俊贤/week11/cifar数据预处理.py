#该文件负责读取Cifar-10数据并对其进行数据增强预处理
import os
import tensorflow as tf
num_classes=10 #图片类型
simple_image = 50000 #样本图
test_image = 10000 #测试图

class cifardatasave(object):
    pass

def cirfaread(orifile):
    result = cifardatasave()
    lable_bytes = 1 
    result.hei_bytes = 32
    result.wei_bytes = 32
    result.deph_bytes = 3
    all_bytes = (result.hei_bytes * result.wei_bytes *result. deph_bytes) + lable_bytes
    sigread_size = tf.FixedLengthRecordReader(record_bytes = all_bytes)
    result.key,value = sigread_size.reader.read(orifile)
    all_bytes = tf.decode_raw(value,tf.uint8)
    result.lable = tf.cast(tf.strided_slice(all_bytes,[0],[lable_bytes]),tf.int32)
    threechnnels = tf.shape(tf.strided_slice(all_bytes,[lable_bytes],all_bytes),
                            [result.deph_bytes,result.hei_bytes,result.wei_bytes])
    result.transimg = tf.transpose(threechnnels,[1,2,0])
    return result

def input(location,banch_size,flag):
    filenames = [os.path.join(location, "data_batch_%d.bin" % i) for i in range(1, 6)]
    read_file = cirfaread(filenames)
    tranfloatimg = tf.cast(read_file.transimg,tf.float32)

    if flag != None:
        cutimg = tf.random_crop(tranfloatimg,[24,24,3])
        flipimg = tf.image.random_flip_left_right(cutimg)
        hightlight = tf.image.random_brightness(flipimg,max_delta=0.8)
        copareimg = tf.image.random_contrast(hightlight,upper=1.8,lower=0.2)
        floatimg = tf.image.per_image_standardization(copareimg)

        floatimg.set_shape([24,24,3])
        read_file.lable.set_shape([1])
        min_queue_examples = int(test_image*0.4)
        images_train, labels_train = tf.train.shuffle_batch([floatimg, read_file.label], batch_size=banch_size,
                                                            num_threads=16,
                                                            capacity=min_queue_examples + 3 * banch_size,
                                                            min_after_dequeue=min_queue_examples,
                                                            )
        # 使用tf.train.shuffle_batch()函数随机产生一个batch的image和label

        return images_train, tf.reshape(labels_train, [banch_size])

    else :
        cutimg = tf.random_crop(tranfloatimg, [24, 24, 3])
        floatimg = tf.image.per_image_standardization(cutimg)
        floatimg.set_shape([24, 24, 3])
        read_file.lable.set_shape([1])
        min_queue_examples = int(simple_image * 0.4)
        images_test, labels_test = tf.train.batch([floatimg, read_file.label],
                                                  batch_size=banch_size, num_threads=16,
                                                  capacity=min_queue_examples + 3 * banch_size)
        # 这里使用batch()函数代替tf.train.shuffle_batch()函数
        return images_test, tf.reshape(labels_test, [banch_size])

