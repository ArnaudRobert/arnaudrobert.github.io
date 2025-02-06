---
layout: page
title: Publications
permalink: /publications/
---
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

<br>
<p>
<img style="float: left;" src="/images/Imperial_shield.png" width="80" height="100"> A. Robert. Beyond Markov Decision Processes: How to Leverage Structure to Build Efficient Reinforcement Learning Algorithms. PhD Thesis.
<details markdown="1">
<summary><i>Abstract</i></summary>
A Markov Decision Process (MDP) describes a general framework for modelling decision-making in an uncertain environment. Over the years, researchers have mainly focused on developing Reinforcement Learning (RL) algorithms to learn how to behave optimally in unknown environments without making explicit assumptions about the existence of underlying structures in the MDP.\\
This general approach has been instrumental to RL's widespread success across various domains. However, when additional structure is present and exploitable, enabling algorithms to leverage it might lead to significant efficiency improvement. This work investigates the benefits of designing RL algorithms that can efficiently leverage these structures. 
Specifically, this work focuses on two distinct types of known latent structures.\\
 First, we consider MDPs that exhibit a hierarchical structure; that is, tasks described by such MDPs can be decomposed into a sequence of sub-tasks. In this context, we provide a lower bound on the sample complexity of hierarchical RL algorithms, which allows us to quantify the potential benefit of hierarchical approaches. We also offer a framework for building hierarchical algorithms that leverage a known hierarchical decomposition. The validity of that framework is supported by theoretical guarantees of its efficiency and empirical evidence that it outperforms its monolithic counterpart whenever a hierarchical structure is present.\\
Second, we consider MDPs that exhibit a graphical structure. The algorithm has access to a graph that encodes the conditional independence between state variables, and the unknown dynamics can be inferred only by local components of the graph. In this context, we provide a posterior sampling-based algorithm that theoretically and empirically outperforms RL algorithms that do not leverage this structural property. Finally, we provide empirical evidence that this latent graphical structure is present in optimising wind farms' yields and demonstrate the efficiency of our algorithm on that particular task.
</details>
</p>
<br>

<br>
<p>
<img style="float: left;" src="/images/hrl_ub.png" width="100" height="100"> A. Robert, G. Drappo, M. Restelli, A. A. Faisal, A. M. Metelli, C. Pike-Burke, Efficient Exploitation of Hierarchical Structure in Sparse Reward Reinforcement Learning. International Conference on Artificial Intelligence and Statistics.
<details markdown="1">
<summary><i>Abstract</i></summary>
We study goal-conditioned Hierarchical Reinforcement Learning (HRL), where a high-level agent instructs sub-goals to a low-level agent.
Under the assumption of a sparse reward function and known hierarchical decomposition, we propose a new algorithm to learn optimal hierarchical policies.
Our algorithm takes a low-level policy as input and is flexible enough to work with a wide range of low-level policies.
We show that when the algorithm that computes the low-level policy is optimistic and provably efficient, our HRL algorithm enjoys a regret bound which represents a significant improvement compared to previous results for HRL. Importantly, our regret upper bound highlights key characteristics of the hierarchical decomposition that guarantee that our hierarchical algorithm is more efficient than the best monolithic approach.
We support our theoretical findings with experiments that underscore that our method consistently outperforms algorithms that ignore the hierarchical structure.
</details>
</p>

<br>
<p>
<img style="float: left;" src="/images/bound_ratio.png" width="100" height="90"> A. Robert, C. Pike-Burke, A. A. Faisal. Sample complexity of goal conditioned hierarchical reinforcement learning. Advances in Neural Information Processing Systems.
<details markdown="1">
<summary><i>Abstract</i></summary>
Hierarchical Reinforcement Learning (HRL) algorithms can perform planning at multiple levels of abstraction. Empirical results have shown that state or temporal abstractions might significantly improve the sample efficiency of algorithms. Yet, we still do not have a complete understanding of the basis of those efficiency gains nor any theoretically grounded design rules. In this paper, we derive a lower bound on the sample complexity for the considered class of goal-conditioned HRL algorithms. The proposed lower bound empowers us to quantify the benefits of hierarchical decomposition and leads to the design of a simple Q-learning-type algorithm that leverages hierarchical decompositions. We empirically validate our theoretical findings by investigating the sample complexity of the proposed hierarchical algorithm on a spectrum of tasks (hierarchical $n$-rooms, Gymnasium's Taxi). The hierarchical $n$-rooms tasks were designed to allow us to dial their complexity over multiple orders of magnitude. Our theory and algorithmic findings provide a step towards answering the foundational question of quantifying the improvement hierarchical decomposition offers over monolithic solutions in reinforcement learning.
</details>
</p>

<br>
<p>
<img style="float: left;" src="/images/windrose.png" width="100" height="80"> S. Li, A. Robert, A. A. Faisal, M. D. Piggott. Learning to optimise wind farms with graph transformers.
<details markdown="1">
<summary><i>Abstract</i></summary>
This work proposes a novel data-driven model capable of providing accurate predictions for the power generation of all wind turbines in wind farms of arbitrary layout, yaw angle configurations and wind conditions. The proposed model functions by encoding a wind farm into a fully connected graph and processing the graph representation through a graph transformer. The resultant graph transformer surrogate demonstrates robust generalisation capabilities and effectively uncovers latent structural patterns embedded within the graph representation of wind farms. The versatility of the proposed approach extends to the optimisation of yaw angle configurations through the application of genetic algorithms. This evolutionary optimisation strategy facilitated by the graph transformer surrogate achieves prediction accuracy levels comparable to industrially standard wind farm simulation tools, with a relative accuracy of more than 99\% in identifying optimal yaw angle configurations of previously unseen wind farm layouts. An additional advantage lies in the significant reduction in computational costs, positioning the proposed methodology as a compelling tool for efficient and accurate wind farm optimisation.
</details>
</p>

<br>
<p>
<img style="float: left;" src="/images/gp.png" width="120" height="50"> F. Tobar, A. Robert, JF Silva. Gaussain process deconvolution.
Procedings of the Royal Society A.
<details markdown="1">
<summary><i>Abstract</i></summary>
Let us consider the deconvolution problem, that is, to recover a latent source $$x(\cdot)$$ from the observations $$\mathbf{y}=[y_1, \cdots, y_N]$$ of a convolution process $$y=x \star h + \eta $$, where $$\eta$$ is an additive noise, the observations in $$\mathbf{y}$$ might have missing parts with respect to $$y$$, and the filter $$h$$ could be unknown. We propose a novel strategy to address this task when $$x$$ is a continuous-time signal: we adopt a Gaussian process (GP) prior on the source $$x$$, which allows for closed-form Bayesian nonparametric deconvolution. We first analyse the direct model to establish the conditions under which the model is well defined. Then, we turn to the inverse problem, where we study i) some necessary conditions under which Bayesian deconvolution is feasible, and ii) to which extent the filter $$h$$ can be learnt from data or approximated for the blind deconvolution case. The proposed approach, termed Gaussian process deconvolution (GPDC) is compared to other deconvolution methods conceptually, via illustrative examples, and using real-world datasets. 
</details>
</p>

<br>
<p>
<img stlye="float: left;" src="/images/wf.png" width="100" height="50"> E. Cazelles, A. Robert, F. Tobar. The Wasserstein-Fourier distance for time series.
IEEE Transactions on Signal Processing.
<details markdown="1">
<summary><i>Abstract</i></summary>
We propose the Wasserstein-Fourier (WF) distance to measure the (dis)similarity between time series by quantifying the displacement of their energy across frequencies. The WF distance operates by calculating the Wasserstein distance between the (normalised) power spectral densities (NPSD) of time series. Yet this rationale has been considered in the past, we fill a gap in the open literature providing a formal introduction of this distance, together with its main properties from the joint perspective of Fourier analysis and optimal transport. As the main aim of this work is to validate WF as a general-purpose metric for time series, we illustrate its applicability on three broad contexts. First, we rely on WF to implement a PCA-like dimensionality reduction for NPSDs which allows for meaningful visualisation and pattern recognition applications. Second, we show that the geometry induced by WF on the space of NPSDs admits a geodesic interpolant between time series, thus enabling data augmentation on the spectral domain, by averaging the dynamic content of two signals. Third, we implement WF for time series classification using parametric/non-parametric classifiers and compare it to other classical metrics. Supported on theoretical results, as well as synthetic illustrations and experiments on real-world data, this work establishes WF as a meaningful and capable resource pertinent to general distance-based applications of time series. 
</details>
</p>

<br>

<p>

<img style="float: left;" src="/images/var_gp.png" width="100" height="100"> A. Robert. A Comparison of Approximation Methods For Gaussian Process Regression. Master thesis. 
<details markdown="1">

<summary><i>Abstract</i></summary>
Gaussian Process Regression (GPR) is a probabilistic model for regression.
Its ability to quantify uncertainty in its predictions makes it a popular choice for regression.
However, GPR is computationally inefficient --- in order to make predictions; it requires $$\mathcal{O}(N^3)$$ computation time and $$\mathcal{O}(N^2)$$ memory, where $$N$$ is the number of observations.\\
This thesis discusses approximation methods for GPR. We compare methods in two different tasks: posterior inference and hyperparameters learning.
We show comparisons on nine different datasets whose sizes range from 300 to 70'000 observations. \\
In the first part of the thesis, we discuss approximation methods for inference. We first compare the following three different approximation methods that use 
a low-rank approximation of the GP kernel:  Nystr\"{o}m approximation, Subset of Regressors (SoR) method and Deterministic Training Conditional (DTC) method. 
Those methods reduce the computational time requirement to $$\mathcal{O}(m^2N)$$ and the memory requirements to $$\mathcal{O}(mN)$$, where $$m$$ is a constant such that $$m < N$$.
Our results show that DTC systematically produces better results than the other two methods. We then compare two methods that use subsets of data: Subset of Data (SoD) and Conjugate Variational Inference (CVI). We show that CVI performs, in general, better than SoD. \\   
In the second part, we compare three different methods for learning: SoD, variational inference for DTC (referred to as varDTC) and stochastic variational inference on DTC (referred to as SVI-GP). The computational time required by varDTC is $$\mathcal{O}(m^2N)$$ while the computational time required by SoD and SVI-GP is $$\mathcal{O}(m^3)$$. We show in our experiments that SVI-GP and SoD have a much lower cost per iteration than varDTC, but SVI-GP is sensitive to initialization and noise in the stochastic gradients.

</details>

</p>


