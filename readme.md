# Machine Learning 2#
Experiments on the plasticity stability dilemma in a continual learning setting 

# Problem #
Our Continual Learning problem our agent needs to differentiate between two 32x32 downsampled Imagenet classes in sequence. 
Our agent only has access to the data for the current task but loses this access as soon as the next task starts.
# Evaluation # 
Stability is often measured by evaluating an agent's ability to perform Task m after training on Task n (where n $>$ m).
However, I propose a different approach: measuring stability based on the time required for the agent to relearn Task m.
This method aligns more closely with how humans retain and recall knowledge.
Humans tend to remember tasks better when they are repeated closer to the initial learning period, but as time passes without repetition, the knowledge fades, requiring more effort to relearn.

# Approach #
My approach to solving this problem is based on the observation that long-term and short-term memory can be modeled as a continuum rather than as two discrete entities.
Most existing algorithms rely on discrete modeling of these brain structures, which may limit their ability to capture the complex interplay between memory types.
 By adopting a continuous modeling perspective, we can explore new ways to address these challenges.\\
The proposed solution involves constructing a standard ANN with a convolutional network as the front end, followed by FC layers.
Rather than applying traditional backpropagation equally across all layers, the approach introduces a gradual reduction in backpropagation's effect on earlier layers as the network encounters later tasks.\\
To achieve this, the standard backpropagation formula is adapted using a hyperparameter \( k \in [0,1] \) and a function as exponent \( f(z, n) \), where \( n \) represents the current task,
 and \( z \) denotes the layer index (counted from back to front). This adaptation enables the implementation of linear, exponential, or other forms of decay.
 The modified backpropagation is expressed as:\\
\[
    bp_{\text{new}} = bp_{\text{standard}} \cdot k^{f(z, n)}, \quad \text{where } f(z, n) \text{ increases with higher } z \text{ and } n.
\]
This strategy emulates long-term memory by allowing earlier layers to encode general, frequently occurring patterns while maintaining adaptability in the later layers through reduced restrictions on parameter updates.
 Consequently, the earlier layers capture stable and recurring features, while the later layers remain flexible to accommodate task-specific variations.

# Results  
