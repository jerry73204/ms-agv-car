#!/usr/bin/env python3
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
# 本程式檔據隨附於套件的的「LICENSE.txt」檔案內容授權，感謝你遵守協議。

import os
import argparse

import tensorflow as tf


def main():
    arg_parser = argparse.ArgumentParser(description='轉換 TensorFlow 模型爲 ncsdk 可讀格式')
    arg_parser.add_argument(
        '--saved-model-dir',
        required=True,
        help='輸入模型目錄',
    )
    arg_parser.add_argument(
        '--output-model-dir',
        required=True,
        help='輸出模型目錄',
    )
    args = arg_parser.parse_args()

    output_prefix = os.path.join(args.output_model_dir, 'model')

    with tf.Session() as sess:
        tf.saved_model.loader.load(
            sess,
            [tf.saved_model.tag_constants.SERVING],
            args.saved_model_dir,
        )
        saver = tf.train.Saver()
        saver.save(sess, output_prefix)

        # with tf.summary.FileWriter('log', sess.graph) as writer:
        #     writer.add_graph(sess.graph)

        # for op in sess.graph.get_operations():
        #     print(op.name)

if __name__ == '__main__':
    main()
