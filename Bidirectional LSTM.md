# Bidirectional LSTM

# **Recurrent Neural Networks**

![Untitled](Bidirectional%20LSTM%20f373fb0a8a9b4b329ec2588b0e1504c2/Untitled.png)

This chain-like nature reveals that recurrent neural networks are intimately related to sequences and lists. They’re the natural architecture of neural network to use for such data.

And they certainly are used! In the last few years, there have been incredible success applying RNNs to a variety of problems: speech recognition, language modeling, translation, image captioning… The list goes on. I’ll leave discussion of the amazing feats one can achieve with RNNs to Andrej Karpathy’s excellent blog post, [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/). But they really are pretty amazing.

Essential to these successes is the use of “LSTMs,” a very special kind of recurrent neural network which works, for many tasks, much much better than the standard version. Almost all exciting results based on recurrent neural networks are achieved with them. It’s these LSTMs that this essay will explore.

# **The Problem of Long-Term Dependencies**

One of the appeals of RNNs is the idea that they might be able to connect previous information to the present task, such as using previous video frames might inform the understanding of the present frame. If RNNs could do this, they’d be extremely useful. But can they? It depends.

Sometimes, we only need to look at recent information to perform the present task. For example, consider a language model trying to predict the next word based on the previous ones. If we are trying to predict the last word in “the clouds are in the *sky*,” we don’t need any further context – it’s pretty obvious the next word is going to be sky. In such cases, where the gap between the relevant information and the place that it’s needed is small, RNNs can learn to use the past information.

![Untitled](Bidirectional%20LSTM%20f373fb0a8a9b4b329ec2588b0e1504c2/Untitled%201.png)

# LSTM NetWorks

[Deep Learning | Introduction to Long Short Term Memory - GeeksforGeeks](https://www.geeksforgeeks.org/deep-learning-introduction-to-long-short-term-memory/)

[https://colah.github.io/posts/2015-08-Understanding-LSTMs/](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

**Long Short Term Memory networks** – usually just called “LSTMs” – are a special kind of RNN, capable of learning long-term dependencies. They were introduced by [Hochreiter & Schmidhuber (1997)](http://www.bioinf.jku.at/publications/older/2604.pdf), and were refined and popularized by many people in following work.[1](https://colah.github.io/posts/2015-08-Understanding-LSTMs/#fn1) They work tremendously well on a large variety of problems, and are now widely used.

LSTMs are explicitly designed to avoid the long-term dependency problem. Remembering information for long periods of time is practically their default behavior, not something they struggle to learn!

All recurrent neural networks have the form of a chain of repeating modules of neural network. In standard RNNs, this repeating module will have a very simple structure, such as a single tanh layer.

![Untitled](Bidirectional%20LSTM%20f373fb0a8a9b4b329ec2588b0e1504c2/Untitled%202.png)

                             **The repeating module in a standard RNN contains a single layer.**

LSTMs also have this chain like structure, but the repeating module has a different structure. Instead of having a single neural network layer, there are four, interacting in a very special way.

![Untitled](Bidirectional%20LSTM%20f373fb0a8a9b4b329ec2588b0e1504c2/Untitled%203.png)

                         **The repeating module in an LSTM contains four interacting layers.**

Don’t worry about the details of what’s going on. We’ll walk through the LSTM diagram step by step later. For now, let’s just try to get comfortable with the notation we’ll be using.

![Untitled](Bidirectional%20LSTM%20f373fb0a8a9b4b329ec2588b0e1504c2/Untitled%204.png)

In the above diagram, each line carries an entire vector, from the output of one node to the inputs of others. The pink circles represent pointwise operations, like vector addition, while the yellow boxes are learned neural network layers. Lines merging denote concatenation, while a line forking denote its content being copied and the copies going to different locations.

# **The Core Idea Behind LSTMs**

The key to LSTMs is the cell state, the horizontal line running through the top of the diagram.

The cell state is kind of like a conveyor belt. It runs straight down the entire chain, with only some minor linear interactions. It’s very easy for information to just flow along it unchanged.

![Untitled](Bidirectional%20LSTM%20f373fb0a8a9b4b329ec2588b0e1504c2/Untitled%205.png)

The LSTM does have the ability to remove or add information to the cell state, carefully regulated by structures called gates.

Gates are a way to optionally let information through. They are composed out of a sigmoid neural net layer and a pointwise multiplication operation.

![Untitled](Bidirectional%20LSTM%20f373fb0a8a9b4b329ec2588b0e1504c2/Untitled%206.png)

The sigmoid layer outputs numbers between zero and one, describing how much of each component should be let through. A value of zero means “let nothing through,” while a value of one means “let everything through!”

An LSTM has three of these gates, to protect and control the cell state.

# **Step-by-Step LSTM Walk Through**

### Forget Gate

![Untitled](Bidirectional%20LSTM%20f373fb0a8a9b4b329ec2588b0e1504c2/Untitled%207.png)

### Input Gate

![Untitled](Bidirectional%20LSTM%20f373fb0a8a9b4b329ec2588b0e1504c2/Untitled%208.png)

![Untitled](Bidirectional%20LSTM%20f373fb0a8a9b4b329ec2588b0e1504c2/Untitled%209.png)

### Output Gate

![Untitled](Bidirectional%20LSTM%20f373fb0a8a9b4b329ec2588b0e1504c2/Untitled%2010.png)

# **Variants on Long Short Term Memory**

![Untitled](Bidirectional%20LSTM%20f373fb0a8a9b4b329ec2588b0e1504c2/Untitled%2011.png)

![Untitled](Bidirectional%20LSTM%20f373fb0a8a9b4b329ec2588b0e1504c2/Untitled%2012.png)

![Untitled](Bidirectional%20LSTM%20f373fb0a8a9b4b329ec2588b0e1504c2/Untitled%2013.png)

# Frequently Asked Questions (FAQs)

### **1. What is LSTM?**

> LSTM is a type of recurrent neural network (RNN) that is designed to address the vanishing gradient problem, which is a common issue with RNNs. LSTMs have a special architecture that allows them to learn long-term dependencies in sequences of data, which makes them well-suited for tasks such as machine translation, speech recognition, and text generation.
> 

### **2. How does LSTM work?**

> LSTMs use a cell state to store information about past inputs. This cell state is updated at each step of the network, and the network uses it to make predictions about the current input. The cell state is updated using a series of gates that control how much information is allowed to flow into and out of the cell.
> 

### **3. What is the major difference between lstm and bidirectional lstm?**

> The vanishing gradient problem of the RNN is addressed by both LSTM and GRU, which differ in a few ways. These distinctions are as follows:Bidirectional LSTM can utilize information from both past and future, whereas standard LSTM can only utilize past info.Whereas GRU only employs two gates, LSTM uses three gates to compute the input of sequence data.Compared to LSTM, GRUs are typically faster and simpler.GRUs are favored for small datasets, while LSTMs are preferable for large datasets.
> 

### **4. What is the difference between LSTM and Gated Recurrent Unit (GRU)?**

> LSTM has a cell state and gating mechanism which controls information flow, whereas GRU has a simpler single gate update mechanism. LSTM is more powerful but slower to train, while GRU is simpler and faster.
> 

### 5. What is difference between LSTM and RNN?

> RNNs have a simple recurrent structure with unidirectional information flow.LSTMs have a gating mechanism that controls information flow and a cell state for long-term memory.LSTMs generally outperform RNNs in tasks that require learning long-term dependencies.
> 

# BiLSTM

[Bidirectional LSTM in NLP - GeeksforGeeks](https://www.geeksforgeeks.org/bidirectional-lstm-in-nlp/)

[**Bidirectional LSTM**](https://www.geeksforgeeks.org/bidirectional-lstm-in-nlp/) (Bi LSTM/ BLSTM) is recurrent neural network (RNN) that is able to process sequential data in both forward and backward directions. This allows Bi LSTM to learn longer-range dependencies in sequential data than traditional LSTMs, which can only process sequential data in one direction.

# BiLSTM **Architecture**

The architecture of bidirectional LSTM comprises of two unidirectional LSTMs which process the sequence in both forward and backward directions. This architecture can be interpreted as having two separate LSTM networks, one gets the sequence of tokens as it is while the other gets in the reverse order. Both of these LSTM network returns a probability vector as output and the final output is the combination of both of these probabilities. It can be represented as:

![Untitled](Bidirectional%20LSTM%20f373fb0a8a9b4b329ec2588b0e1504c2/Untitled%2014.png)