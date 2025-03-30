# Machine Learning 2: Experiments on the Plasticity-Stability Dilemma in Continual Learning  

## 📌 Problem  
In our continual learning setup, an agent must sequentially differentiate between two downsampled 32×32 ImageNet classes. The agent has access only to the current task's data, losing it as soon as the next task begins.  
<p float="left">
  <img src="images\readme\cl_problem.png" width="99%" />
</p>

## 📊 Evaluation  
Traditionally, stability is measured by evaluating an agent's performance on Task *m* after training on Task *n* (where *n > m*). However, I propose a different approach:  

🔹 **Measuring stability based on the time required to relearn Task *m*** rather than just assessing its retention.  

This approach aligns with how humans retain and recall knowledge. Humans tend to remember tasks better when they are repeated closer to the initial learning period. However, as time passes without repetition, knowledge fades, requiring more effort to relearn.  

## 🏗 Approach  
Our approach is inspired by the idea that long-term and short-term memory exist on a **continuum** rather than as discrete entities. Most existing algorithms model these memory structures as separate, limiting their ability to capture the complex interplay between memory types.  

### 🔬 Key Idea  
We construct a **standard artificial neural network (ANN)** with a convolutional network as the front end, followed by fully connected (FC) layers. Instead of applying **uniform backpropagation** across all layers, we introduce a gradual **reduction** in backpropagation’s effect on earlier layers as the network encounters new tasks.  

### ⚙️ Modified Backpropagation  
We modify the standard backpropagation formula using a hyperparameter **\( k \in [0,1] \)** <br />
and an exponent function **\( f(z, n) \)**, where:  
- **\( n \)** is the current task index.  
- **\( z \)** is the layer index (counted from back to front).  

This enables us to implement **linear, exponential, or other decay functions** in backpropagation.  
where **\( f(z, n) \)** increases with **\( z \)** and **\( n \)**.  

### 🧠 Intuition  
This approach **mimics human memory** by allowing:  
✅ **Earlier layers** to capture general, frequently occurring patterns (long-term memory).  
✅ **Later layers** to remain adaptable and flexible for task-specific details (short-term memory).  
<p float="left">
  <img src="Evaluation\images\Baselines_Curriculum_Completions.png" width="49%" />
  <img src="Evaluation\images\Baselines_Curriculum_Scores.png" width="49%" />
</p>

## 📈 Results  
📌 *To be added...*  
