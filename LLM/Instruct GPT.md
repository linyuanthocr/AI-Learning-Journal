# Instruct GPT

![Untitled](images/Instruct%20GPT%20485a15aef4474679bda9a098b870f6e0/Untitled.png)

We focus on fine-tuning approaches to aligning language models. Specifically, we use reinforcement learning from human feedback (RLHF; Christiano et al., 2017; Stiennon et al., 2020) to fine-tune GPT-3 to follow a broad class of written instructions (see Figure 2). This technique uses human preferences as a reward signal to fine-tune our models. We first hire a team of 40 contractors to label our data, based on their performance on a screening test (see Section 3.4 and Appendix B.1 for more details). We then collect a dataset of human-written demonstrations of the desired output behavior on (mostly English) prompts submitted to the OpenAI API3 and some labeler-written prompts, and use this to train our supervised learning baselines. Next, we collect a dataset of **human-labeled comparisons** between outputs from our models on a larger set of API prompts. We then train a reward model (RM) on this dataset to predict which model output our labelers would prefer. Finally, we use this RM as a reward function and fine-tune our supervised learning baseline to maximize this reward using the PPO algorithm (Schulman et al., 2017).

We train **three model** sizes (1.3B, 6B, and 175B parameters), and all of our models use the GPT-3 architecture.

### 3 **Datasets**

![Untitled](images/Instruct%20GPT%20485a15aef4474679bda9a098b870f6e0/Untitled%201.png)

### Methodology

We start with a pretrained language model, a distribution of prompts on which we want our model to produce aligned outputs, and a team of trained human labelers (see Sections 3.4 for details). We then apply the following three steps (Figure 2).
**Step 1: Collect demonstration data, and train a supervised policy**. Our labelers provide demon- strations of the desired behavior on the input prompt distribution (see Section 3.2 for details on this distribution). We then **fine-tune a pretrained GPT-3 model** on this data using supervised learning.
**Step 2: Collect comparison data, and train a reward model**. We collect a dataset of comparisons between model outputs, where labelers indicate which output they prefer for a given input. We then train a reward model to predict the human-preferred output.
**Step 3: Optimize a policy against the reward model using PPO.** We use the output of the RM as a scalar reward. We fine-tune the supervised policy to optimize this reward using the PPO algorithm (Schulman et al., 2017).
Steps 2 and 3 can be iterated continuously; more comparison data is collected on the current best policy, which is used to train a new RM and then a new policy. In practice, most of our comparison data comes from our supervised policies, with some coming from our PPO policies.

### **Models**

We start with the **GPT-3 pre-trained language models,**  train models with three different techniques:

1. **Supervised fine-tuning (SFT)**. We fine-tune GPT-3 on our labeler demonstrations using supervised learning.
2. **Reward modeling (RM).** Starting from **the SFT model** with the final unembedding layer removed, we trained a model to take in a prompt and response, and output a scalar reward. In this paper we only use 6B RMs, as this saves a lot of compute, and we found that 175B RM training could be unstable. **Pairwise ranking loss**

![Untitled](images/Instruct%20GPT%20485a15aef4474679bda9a098b870f6e0/Untitled%202.png)

1. **Reinforcement learning (RL).** We fine-tuned **the SFT model** on our environment using PPO (Schulman et al., 2017). The environment is a bandit environment which presents *a random customer prompt and expects a response to the prompt*. Given the prompt and response, it produces a reward determined by the reward model and ends the episode. In addition, we add **a per-token KL penalty from the SFT model** at each token to mitigate over- optimization of the reward model. The value function is initialized from the RM. We call these models “**PPO**.”
We also experiment with mixing the pretraining gradients into the PPO gradients, in order to fix the performance regressions on public NLP datasets. We call these models “**PPO-ptx**.” We maximize the following combined objective function in RL training:

![Untitled](images/Instruct%20GPT%20485a15aef4474679bda9a098b870f6e0/Untitled%203.png)

PPO-ptx 是利用学好的RM对于SFT的再训练，训练后的模型就是InstructGPT
