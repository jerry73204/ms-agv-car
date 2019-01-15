#!/usr/bin/env python3
import argparse
import time

import cv2
import numpy as np
from openvino.inference_engine import IENetwork, IEPlugin


def main():
    # 設定程式參數
    arg_parser = argparse.ArgumentParser(description='使用 Movidius 進行預測')
    arg_parser.add_argument(
        '--model-file',
        required=True,
        help='模型架構檔',
    )
    arg_parser.add_argument(
        '--weights-file',
        required=True,
        help='模型參數檔',
    )
    arg_parser.add_argument(
        '--video-type',
        choices=['file', 'camera'],
        default='camera',
        help='影片類型',
    )
    arg_parser.add_argument(
        '--source',
        default='/dev/video0',
        help='影片來源檔',
    )
    arg_parser.add_argument(
        '--input-width',
        type=int,
        default=48,
        help='模型輸入影像寬度',
    )
    arg_parser.add_argument(
        '--input-height',
        type=int,
        default=48,
        help='模型輸入影像高度',
    )
    arg_parser.add_argument(
        '--gui',
        action='store_true',
        help='啓用圖像界面',
    )

    # 解讀程式參數
    args = arg_parser.parse_args()
    assert args.input_width > 0 and args.input_height > 0

    # 設置 Movidius 裝置
    plugin = IEPlugin(device='MYRIAD')

    # 載入模型檔
    net = IENetwork.from_ir(model=args.model_file, weights=args.weights_file)
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
    exec_net = plugin.load(network=net)

    # 開啓影片來源
    if args.video_type == 'file':  # 檔案
        video_dev = cv2.VideoCapture(args.source)
        video_width = video_dev.get(cv2.CAP_PROP_FRAME_WIDTH)
        video_height = video_dev.get(cv2.CAP_PROP_FRAME_HEIGHT)

    elif args.video_type == 'camera':  # 攝影機
        video_dev = cv2.VideoCapture(0)

    # 主迴圈
    try:
        prev_timestamp = time.time()

        while True:
            ret, orig_image = video_dev.read()
            curr_time = time.localtime()

            # 檢查串流是否結束
            if ret is None or orig_image is None:
                break

            # 縮放爲模型輸入的維度、調整數字範圍爲 0～1 之間的數值
            preprocessed_image = cv2.resize(
                orig_image.astype(np.float32),
                (args.input_width, args.input_height),
            ) / 255.0

            # 這步驟打包圖片成大小爲 1 的 batch
            batch = np.expand_dims(
                np.transpose(preprocessed_image, (2, 0 ,1)),  # 將維度順序從 NHWC 調整爲 NCHW
                0,
            )

            # 執行預測
            request_handle = exec_net.start_async(
                request_id=0,
                inputs={input_blob: batch}
            )
            status = request_handle.wait()
            result_batch = request_handle.outputs[out_blob]
            result_onehot = result_batch[0]

            # 判定結果
            left_score, right_score, stop_score, other_score = result_onehot
            class_id = np.argmax(result_onehot)

            if class_id == 0:
                class_str = 'left'
            elif class_id == 1:
                class_str = 'right'
            elif class_id == 2:
                class_str = 'stop'
            elif class_id == 3:
                class_str = 'other'

            # 計算執行時間
            recent_timestamp = time.time()
            period = recent_timestamp - prev_timestamp
            prev_timestamp = recent_timestamp

            print('時間：%02d:%02d:%02d ' % (curr_time.tm_hour, curr_time.tm_min, curr_time.tm_sec))
            print('輸出：%.2f %.2f %.2f %.2f' % (left_score, right_score, stop_score, other_score))
            print('類別：%s' % class_str)
            print('費時：%f' % period)
            print()

            # 顯示圖片
            if args.gui:
                cv2.imshow('', orig_image)
                cv2.waitKey(1)

    except KeyboardInterrupt:
        print('使用者中斷')

    # 終止影像裝置
    video_dev.release()


if __name__ == '__main__':
    main()
