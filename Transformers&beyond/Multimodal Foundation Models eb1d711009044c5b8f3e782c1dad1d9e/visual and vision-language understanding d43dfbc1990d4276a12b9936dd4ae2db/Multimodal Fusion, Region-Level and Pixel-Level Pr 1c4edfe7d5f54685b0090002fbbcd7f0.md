# Multimodal Fusion, Region-Level and Pixel-Level Pre-training

These methods typically use a pre-trained image encoder at the first hand to perform a second-stage pre-training. 

# From Multimodal Fusion to Multimodal LLM

For dual encoders such as CLIP (Radford et al., 2021), image and text are encoded separately, and modality interaction is only handled via a simple dot product of image and text feature vectors. This can be very effective for zero-shot image classification and image-text retrieval. However, due to the **lack of deep multimodal fusion**, CLIP alone performs poorly on the image captioning (Vinyals et al., 2015) and visual question answering (Antol et al., 2015) tasks. This requires the pre-training of a fusion encoder, where additional transformer layers are typically employed to model the deep interaction between image and text representations. Below, we review how these fusion-encoder pre-training methods are developed over time

### Region-Level Pre-training

![Untitled](Multimodal%20Fusion,%20Region-Level%20and%20Pixel-Level%20Pr%201c4edfe7d5f54685b0090002fbbcd7f0/Untitled.png)

Object detection contains two sub-tasks: localization and recognition

Specifically, ViLD (Gu et al., 2021) and RegionCLIP (Zhong et al., 2022a) distill knowledge from CLIP with a two-stage detector for zero-shot object detection. In MDETR (Kamath et al., 2021) and GLIP (Li et al., 2022e) (as shown in Figure 2.14), the authors propose to reformulate detection as a phrase grounding problem, and perform grounded language-image pre-training. GLIPv2 (Zhang et al., 2022b) and FIBER (Dou et al., 2022a) further perform unified pre-training for both grounding and vision-language understanding tasks.

### Pixel-Level Pre-training

The Segment Anything Model (SAM) (Kirillov et al., 2023) is a recent vision foundation model for image segmentation that aims to perform pixel-level pre-training. 

![Untitled](Multimodal%20Fusion,%20Region-Level%20and%20Pixel-Level%20Pr%201c4edfe7d5f54685b0090002fbbcd7f0/Untitled%201.png)

As depicted in Figure 2.15, the objective of the Segment Anything project is to develop a founda- tional vision model for segmentation. This model is designed to be readily adaptable to a wide range of both existing and novel segmentation tasks, such as edge detection, object proposal generation, instance segmentation, open-vocabulary segmentation, and more

User-friendly approach:

**Task**. The authors propose the promptable segmentation task, where the goal is to return a valid segmentation mask given any segmentation **prompt**, such as **a set of points, a rough box or mask, or free-form text**.

**Model**. The architecture of SAM is conceptually simple. It is composed of three main com- ponents: (i) a powerful **image** **encoder** (MAE (He et al., 2022a) pre-trained ViT); (ii) a **prompt encoder** (for sparse input such as points, boxes, and free-form text, the CLIP text encoder is used; for dense input such as masks, a convolution operator is used); and (iii) **a lightweight mask decoder** based on transformer.
**Data**. To acquire large-scale data for pre-training, the authors develop a data engine that performs **model-in-the-loop dataset** **annotation**