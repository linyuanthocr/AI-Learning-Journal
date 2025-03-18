# Humanoid robot

# HElix

https://www.figure.ai/news/helix

![image.png](images/Humanoid%20robot%201ae71bdab3cf80fe9590f1a8bdf394ed/image.png)

# Model and Training Details

**Data**

We collect a high quality, multi-robot, multi-operator dataset of diverse teleoperated behaviors, ~500 hours in total. To generate natural language-conditioned training pairs, we use an auto-labeling VLM to generate hindsight instructions. The VLM processes segmented video clips from the onboard robot cameras, prompted with: "What instruction would you have given the robot to get the action seen in this video?" All items handled during training are excluded from evaluations to prevent contamination.

**Architecture**

Our system comprises two main components: S2, a VLM backbone, and S1, a latent-conditional visuomotor transformer. S2 is built on a 7B-parameter open-source, open-weight VLM pretrained on internet-scale data. It processes monocular robot images and robot state information (consisting of wrist pose and finger positions) after projecting them into vision-language embedding space. Combined with natural language commands specifying desired behaviors, S2 distills all semantic task-relevant information into a single continuous latent vector, passed to S1 to condition its low-level actions.

S1, an 80M parameter cross-attention encoder-decoder transformer, handles low-level control. It relies on a fully convolutional, multi-scale vision backbone for visual processing, initialized from pretraining done entirely in simulation. While S1 receives the same image and state inputs as S2, it processes them at a higher frequency to enable more responsive closed-loop control. The latent vector from S2 is projected into S1's token space and concatenated with visual features from S1's vision backbone along the sequence dimension, providing task conditioning.

S1 outputs full upper body humanoid control at 200hz, including desired wrist poses, finger flexion and abduction control, and torso and head orientation targets. We append to the action space a synthetic "percentage task completion" action, allowing Helix to predict its own termination condition, which makes it easier to sequence multiple learned behaviors.

**Training**

Helix is trained fully end-to-end, mapping from raw pixels and text commands to continuous actions with a standard regression loss. Gradients are backpropagated from S1 into S2 via the latent communication vector used to condition S1's behavior, allowing joint optimization of both components. Helix requires no task-specific adaptation; it maintains a single training stage and single set of neural network weights without separate action heads or per-task fine-tuning stages.

During training, we add a temporal offset between S1 and S2 inputs. This offset is calibrated to match the gap between S1 and S2's deployed inference latency, ensuring that the real-time control requirements during deployment are accurately reflected in training.

**Optimized Streaming Inference**

Helix's training design enables efficient model parallel deployment on Figure robots, each equipped with dual low-power-consumption embedded GPUs. The inference pipeline splits across S2 (high-level latent planning) and S1 (low-level control) models, each running on dedicated GPUs. S2 operates as an asynchronous background process, consuming the latest observation (onboard camera and robot state) and natural language commands. It continuously updates a shared memory latent vector that encodes the high-level behavioral intent.

S1 executes as a separate real-time process, maintaining the critical 200Hz control loop required for smooth whole upper body action. It takes both the latest observation and the most recent S2 latent vector. The inherent speed difference between S2 and S1 inference naturally results in S1 operating with higher temporal resolution on robot observations, creating a tighter feedback loop for reactive control.

This deployment strategy deliberately mirrors the temporal offset introduced in training, minimizing the train-inference distribution gap. The asynchronous execution model allows both processes to run at their optimal frequencies, allowing us to run Helix as fast as our fastest single task imitation learning policies.
