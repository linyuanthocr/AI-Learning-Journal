# LightGlue

![Untitled](images/LightGlue%2077c420ad7455441e94b437afa603ae44/Untitled.png)

![Untitled](images/LightGlue%2077c420ad7455441e94b437afa603ae44/Untitled%201.png)

LightGlue is made of a stack of L identical layers that process the two sets jointly. Each layer is composed of self- and cross-attention units that update the representation of each point. A classifier then decides, at each layer, whether to halt the inference, thus avoiding unnecessary computations. A lightweight head finally computes a partial assignment from the set of representations.

1. Initialize the feature state with descriptor, and update this state with LightGlue by each layer (one self attention and one cross attention). 
2. Attention unit

![Untitled](images/LightGlue%2077c420ad7455441e94b437afa603ae44/Untitled%202.png)

**Attention score:**

![Untitled](images/LightGlue%2077c420ad7455441e94b437afa603ae44/Untitled%203.png)

![Untitled](images/LightGlue%2077c420ad7455441e94b437afa603ae44/Untitled%204.png)

![Untitled](images/LightGlue%2077c420ad7455441e94b437afa603ae44/Untitled%205.png)

**Correspondence prediction**

![Untitled](images/LightGlue%2077c420ad7455441e94b437afa603ae44/Untitled%206.png)

![Untitled](images/LightGlue%2077c420ad7455441e94b437afa603ae44/Untitled%207.png)

### Adaptive depth and width

i) we reduce the number of layers depending on the difficulty of the input image pair;
ii) we prune out points that are confidently rejected early.

**Confidence classifier**

![Untitled](images/LightGlue%2077c420ad7455441e94b437afa603ae44/Untitled%208.png)

![Untitled](images/LightGlue%2077c420ad7455441e94b437afa603ae44/Untitled%209.png)

![Untitled](images/LightGlue%2077c420ad7455441e94b437afa603ae44/Untitled%2010.png)

**Point pruning:** When the exit criterion is not met, points that are predicted as both confident and unmatchable are removed.

# Supervision

we first train it to predict correspondences and only after train the confidence classifier. The latter thus does not impact the accuracy at the final layer or the convergence of the training.

![Untitled](images/LightGlue%2077c420ad7455441e94b437afa603ae44/Untitled%2011.png)

![Untitled](images/LightGlue%2077c420ad7455441e94b437afa603ae44/Untitled%2012.png)

### Comparison with SuperGlue LightGlue

**Positional encoding**: SuperGlue encodes the absolute point positions with an MLP and fuses them early with the descriptors. . LightGlue instead relies on a relative encoding that is better comparable across images and is added in each self-attention unit. 

**Prediction head**: SuperGlue predicts an assignment by solving a differentiable optimal transport problem using the Sinkhorn algorithm [66, 48]. It consists in many iterations of row-wise and column-wise normalization, which is expen- sive in terms of both compute and memory. SuperGlue adds a dustbin to reject unmatchable points. We found that the dustbin entangles the similarity score of all points and thus yields suboptimal training dynamics. LightGlue disentangles similarity and matchability, which are much more efficient to predict. This also yields cleaner gradient

**Deep supervision:** Because of how expensive Sinkhorn is, SuperGlue cannot make predictions after each layer and is supervised only at the last one. ****The lighter head of LightGlue makes it possible to predict an assignment at each layer and to supervise it**.** This speeds up the convergence and enables exiting the inference after any layer, which is key to the efficiency gains of LightGlue
