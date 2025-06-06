# NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis

https://arxiv.org/abs/2003.08934

![image.png](images/NeRF%20Representing%20Scenes%20as%20Neural%20Radiance%20Fields%201b571bdab3cf804a9f26df57cb745cf0/image.png)

Views are synthesized by querying the **5D coordinates** along the camera rays and using **classical volume rendering** techniques to project the output **colors and densities** onto an image

### **Contributions:**

1. An approach for representing continuous scenes with complex geometry and materials as **5D neural radiance fields**, parameterized as basic MLP networks.
2. A **differentiable rendering** procedure based on classical volume rendering techniques, which we use to optimize these representations from standard RGB images. This includes **a hierarchical sampling strategy** to allocate the MLP’s capacity towards space with visible scene content.
    1. classical volume rendering
        
        ![image.png](images/NeRF%20Representing%20Scenes%20as%20Neural%20Radiance%20Fields%201b571bdab3cf804a9f26df57cb745cf0/image%201.png)
        
    2. stratified sampling + quadrature rule:
        
        ![image.png](images/NeRF%20Representing%20Scenes%20as%20Neural%20Radiance%20Fields%201b571bdab3cf804a9f26df57cb745cf0/image%202.png)
        
        Although we use a discrete set of samples to estimate the integral, stratified sampling enables us to **represent a continuous scene** representation because it results in the MLP being **evaluated at continuous positions** over the course of optimization.
       
        ```python
        import torch
        
        def volume_render_radiance_field(sigmas, rgbs, deltas):
            """
            Args:
                sigmas: [N_rays, N_samples] - predicted densities at each sample
                rgbs:   [N_rays, N_samples, 3] - predicted colors at each sample
                deltas: [N_rays, N_samples] - distance between adjacent samples
            Returns:
                final_colors: [N_rays, 3] - rendered RGB for each ray
            """
        
            # 1. Compute alpha = 1 - exp(-sigma * delta)
            alphas = 1.0 - torch.exp(-sigmas * deltas)
        
            # 2. Compute transmittance T_i = cumprod(1 - alpha) with shifting
            # Add a small epsilon to prevent log(0)
            eps = 1e-10
            transmittance = torch.cumprod(torch.cat([
                torch.ones_like(alphas[:, :1]),  # T_0 = 1
                1.0 - alphas + eps
            ], dim=-1), dim=-1)[:, :-1]  # Shift right
        
            # 3. Weights = T_i * alpha_i
            weights = transmittance * alphas  # [N_rays, N_samples]
        
            # 4. Final RGB = weighted sum of colors
            final_colors = torch.sum(weights.unsqueeze(-1) * rgbs, dim=1)  # [N_rays, 3]
        
            return final_colors
        
        ```
        
        ![image.png](images/NeRF%20Representing%20Scenes%20as%20Neural%20Radiance%20Fields%201b571bdab3cf804a9f26df57cb745cf0/image%203.png)
        
3. **A positional encoding** to map each input 5D coordinate into a higher dimensional space, which enables us to successfully optimize neural radiance fields to represent high-frequency scene content
    1. positional encoding:
        - **mapping the inputs to a higher dimensional** space using high frequency functions before passing them to the network enables better fitting of data that contains high frequency variation
        
        ![image.png](images/NeRF%20Representing%20Scenes%20as%20Neural%20Radiance%20Fields%201b571bdab3cf804a9f26df57cb745cf0/image%204.png)
       <img width="881" alt="image" src="https://github.com/user-attachments/assets/89b9568a-1455-4267-98cf-6dfb8a84f120" />
        
    3. Hierarchical volume sampling
        
        ![image.png](images/NeRF%20Representing%20Scenes%20as%20Neural%20Radiance%20Fields%201b571bdab3cf804a9f26df57cb745cf0/image%205.png)
        
    4. Details
        - Input: RGB, intrinsic parameters, camera poses, scene bound (synthetic data and real data (COLMAP SFM estimation))
        - each iteration: random camera rays+hierachical rendering
            
            ![image.png](images/NeRF%20Representing%20Scenes%20as%20Neural%20Radiance%20Fields%201b571bdab3cf804a9f26df57cb745cf0/image%206.png)
            

### **multilayer perceptron (MLP)**

![image.png](images/NeRF%20Representing%20Scenes%20as%20Neural%20Radiance%20Fields%201b571bdab3cf804a9f26df57cb745cf0/image%207.png)

[https://github.com/bmild/nerf](https://github.com/bmild/nerf)

Time cost

train：a single NVIDIA V100 GPU (about 1–2 days)

batch size of 4096 rays, each sampled at Nc = 64
coordinates in the coarse volume and Nf = 128 additional coordinates in the fine volume. 

predict：

requires 640k rays per image, and our real scenes require 762k rays per
image, resulting in between 150 and 200 million network queries per rendered
image. On an NVIDIA V100, this takes approximately 30 seconds per frame.
