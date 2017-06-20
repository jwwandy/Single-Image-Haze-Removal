# Single Image Haze Removal
Implementation of KaiMing He CVPR09 Best Paper <br>
Replace Laplacian Matrix part by KaiMing He ECCV10 Guided Filter
* [Single Image Haze Removal using Dark Channel Prior](http://kaiminghe.com/publications/cvpr09.pdf)
* [Guided Image Filtering](http://kaiminghe.com/publications/eccv10guidedfilter.pdf)


## Requirements
* opencv3
* opencv3-contrib
* skimage
* numpy
* matplotlib


pip installable package in requirements.txt(except opencv), please use
```
pip install requirements.txt
```

## Usage
See the command output
```
python main.py --help
```

## Result

Original|Dark Channel|Raw Transmission Map|Transmission Map After Guided Filter|Output|Output with histrogram equalization
---|---|---|---|---|---
![](./image/forest.jpg)|![](./result/forest/dark.jpg)|![](./result/forest/raw_transmission.jpg)|![](./result/forest/refine_transmission.jpg)|![](./result/forest/noequalize.jpg)|![](./result/forest/equalize.jpg)
![](./image/city.jpg)|![](./result/city/dark.jpg)|![](./result/city/raw_transmission.jpg)|![](./result/city/refine_transmission.jpg)|![](./result/city/noequalize.jpg)|![](./result/city/equalize.jpg)
![](./image/tiananmen.png)|![](./result/tiananmen/dark.jpg)|![](./result/tiananmen/raw_transmission.jpg)|![](./result/tiananmen/refine_transmission.jpg)|![](./result/tiananmen/noequalize.jpg)|![](./result/tiananmen/equalize.jpg)
