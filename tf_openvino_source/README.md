## 使用 TensorFlow + Movidius 進行路牌分類

本範例使用 TensorFlow 訓練模型，資料目錄請依照下面方式編排。訓練用圖片分類成 `left`、`right`、`stop` 目錄，非上述三類放置在 `other`目錄。測試資料置於 `test` 目錄。

```
data_dir
├── left
│   ├── 000.jpg
│   ├── 001.jpg
│   └── ...
├── right
│   ├── 000.jpg
│   ├── 001.jpg
│   └── ...
├── stop
│   ├── 000.jpg
│   ├── 001.jpg
│   └── ...
├── other
│   ├── 000.jpg
│   ├── 001.jpg
│   └── ...
└── test
    ├── 000.jpg
    ├── 001.jpg
    └── ...
```

執行 `train_tensorflow_model.py` 訓練模型，指令詳細說明請參見 `-h` 選項。

上層的 `models/tf_openvino_model` 目錄也有訓練好的模型可參考。

```sh
# 訓練 128 回合，完成輸出模型到 tf_model 目錄
./train_tensorflow_model.py --model-base-dir tf_model \
                            --data-dir ~/dataset
```

使用 Intel OpenVINO 工具的 mo\_tf.py 將模型編譯為 Movidius 模型檔。

```sh
# 設定環境
source /opt/intel/computer_vision_sdk/bin/setupvars.sh

# 路徑中的 XXXXXXXXXX 請依據實際路徑填寫全部都是數字。指令完成時會輸出 `mo2_model` 目錄
mo_tf.py --saved_model_dir tf_model/XXXXXXXXXX \
         --output_dir mo2_model \
         --input_shape "[1,48,48,3]" \
         --input input_image \
         --output probabilities \
         --data_type FP16
```

講模型檔 model.graph 移到 RPi 上，在上面執行 `movidius_video.py` 使用攝影機測試模型。

```sh
./movidius_video.py --model-file mo2_model/tf_model.xml \
                    --weights-file mo2_model/tf_model.bin
```

`movidius_car.py` 是結合軌跡車和 Movidius 模型的範例。

```sh
./movidius_car.py --model-file model.graph
```
