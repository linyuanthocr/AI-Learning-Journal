# Stylebrush

pipeline:

![image.png](Stylebrush%20bd9a4a57c7574ae2b54776ba6209fcc6/image.png)

Training process:

![image.png](Stylebrush%20bd9a4a57c7574ae2b54776ba6209fcc6/image%201.png)

"StyleBrush: Style Extraction and Transfer from a Single Image" by Wancheng Feng and colleagues, published in August 2024.

**Overview:**
StyleBrush introduces a method for transferring the style from a single reference image to other visual content. The architecture comprises two main components:

- **ReferenceNet**: Extracts style features from the reference image.
- **Structure Guider**: Extracts structural features from the input image, facilitating image-guided stylization.

**Key Contributions:**

- **Dataset Creation**: The authors utilized large language models (LLMs) and text-to-image (T2I) models to create a dataset of 100,000 high-quality style images, encompassing diverse styles and content with high aesthetic scores.
- **Training Methodology**: Training pairs were constructed by cropping different regions of the same image, enabling effective learning of style and structure separation.
- **Performance**: Experiments demonstrated that StyleBrush achieves state-of-the-art results in both qualitative and quantitative analyses.

This approach offers a significant advancement in stylization tasks, allowing for accurate style transfer from a single reference image while preserving the structural integrity of the content image.