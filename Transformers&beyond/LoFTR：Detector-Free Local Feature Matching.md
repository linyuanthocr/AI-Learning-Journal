# LoFTR：Detector-Free Local Feature Matching with Transformers

We present a novel method for local image feature matching. Instead of performing image feature detection, description, and matching sequentially, we propose to first establish pixel-wise dense matches at a coarse level and later refine the good matches at a fine level. In contrast to dense methods that use a cost volume to search corre- spondences, we use selfand cross attention layers in Trans- former to obtain feature descriptors that are conditioned on both images. The global receptive field provided by Trans- former enables our method to produce dense matches in low-texture areas, where feature detectors usually strug- gle to produce repeatable interest points. The experiments on indoor and outdoor datasets show that LoFTR outper- forms state-of-the-art methods by a large margin. LoFTR also ranks first on two public benchmarks of visual localization among the published methods. Code is available at our project page: [https://zju3dv.github.io/loftr/](https://zju3dv.github.io/loftr/).

![Untitled](images/LoFTR%EF%BC%9ADetector-Free%20Local%20Feature%20Matching%20with%20Tr%2047df75340294468782c9341493122bd9/Untitled.png)

# Efficient LoFTR

![Untitled](images/LoFTR%EF%BC%9ADetector-Free%20Local%20Feature%20Matching%20with%20Tr%2047df75340294468782c9341493122bd9/Untitled%201.png)

![Untitled](images/LoFTR%EF%BC%9ADetector-Free%20Local%20Feature%20Matching%20with%20Tr%2047df75340294468782c9341493122bd9/Untitled%202.png)
