---
layout: post
title: "Bandits, Rock, Paper and Scissors"
---

This post aims to formalize the bandit problem and employ the EXP3 algorithm to play the game of Rock-Paper-Scissors. While it doesn't adhere to a conventional textbook approach, as it utilizes an adversarial algorithm within a stochastic environment, which is itself adversarial by nature, the post was valuable in organizing ideas and enhancing comprehension of the bandit problem.

#### Initial Setup
Rock-Paper-Scissors (RPS) is a simple game that serves as a good environment to implement some learning algorithms. Two players simultaneously choose from three options: rock, paper, or scissors. The winner is determined by the following rules:

`- Rock beats scissors`\
`- Scissors beats paper`\
`- Paper beats rock`

We can model this game in the bandit framework. A bandit problem can be defined as follows:

**Definition 1 (Bandit Problem)** A bandit problem is a tuple $(\mathcal{A} ,\mathcal{C}, \mathcal{R})$, where $\mathcal{A}$ is the set of arms (or actions), $\mathcal{C}$ is the set of contexts and $\mathcal{R}: \mathcal{A} \times \mathcal{C} \to \Delta(\mathbb{R})$ is the reward function, where $\Delta(\mathbb{R})$ is the space of probability distributions over $\mathbb{R}$. 

This is a sequential decision problem where a learner interacts with an environment over $n$ rounds, $n \in \mathbb{N}$. In each round $t \in [n]$ the learner receives a context $c_{t} \in \mathcal{C}$, that can be seen as the state of the environment, chooses an action $a_{t} \in \mathcal{A}$ and the environment then reveals a reward $x_{t} \sim \mathcal{R}(a_{t}, c_{t})$.  

The context space can be unitary, resulting in the irrelevance of the context for the agent's decision-making. Thus, we can define $\mathcal{R}(a,c) = \mathcal{R}(a)$, for $c \in \mathcal{C}$ and every $a \in \mathcal{A}$, incorporating the only context into the law of the rewards. This occurs in the classic slot machine example, where the learner has $K$ arms to choose from, and the context doesn't matter. On the other hand, suppose we were to play the same $K$ slot machines in different geographical regions. In that case, we would have different contexts that could affect the reward (the new owner could modify the reward mechanism, for example).

Another remark about the context is that $\mathbb{P}(c_{t+1}\vert c_{t},a_{t}) = \mathbb{P}(c_{t+1}\vert c_{t})$, for every $t \in [n]$. This means that the learner has no influence over the context dynamics. This contrasts with the Reinforcement Learning (RL) framework, where the learner can influence the state dynamics through her actions. 

The reward associated with the action chosen are always revealed and the rewards associated with the other actions not chosen may or may not be revealed to the learner. The later, when revealed, is an important information to the learner, since she can make a better update of the estimated reward distribution for each arm, even if she never play most of them.

The way the reward function behaves determines the type of bandit problem that the learner is facing. If the reward function remains the same regardless of the learner's actions, then the learner is facing a stochastic bandit problem. One example of this kind is a play of $K$ slot machines. On the other hand, if the reward function reacts to the learner's actions, then the learner is facing an adversarial bandit problem. A common way to see an adversarial bandit is when, using a thought experiment, the adversary knows in advance each action of the learner and chooses the reward path in a way that decreases the learner's total reward. This is not an intuitive way to capture the nature of the adversarial bandit since such a scenario doesn't happen in real life. A natural way that we could see such a problem would be from a game-theoretical one, where at each round, the adversary adapts its policy to new information that it has just gathered. But as pointed out by [1], this is a similar but not the same problem as the adversary commonly presented in textbooks. Regardless, in this post, we will use the adversarial bandit problem in a game-theoretical framework as a way to model the RPS game.

Returning to the RPS game, we set the tuple $(\mathcal{A}, \mathcal{C}, \mathcal{R})$ as follows: $\mathcal{A} = \\{R,P,S\\}$, $\mathcal{C} = \{I_{1}, I_{2}, \dots\}$, the set of different adversaries, and the reward functions is defined as follows:

$$
\begin{align*}
\mathcal{R}_{t}(R,I_{j}) = [\pi_{R,t}^{j}(0),\pi_{P,t}^{j}(-1),\pi_{S,t}^{t}(1)]\\
\mathcal{R}_{t}(P,I_{j}) = [\pi_{R,t}^{j}(1),\pi_{P,t}^{j}(0),\pi_{S,t}^{j}(-1)]\\
\mathcal{R}_{t}(S,I_{j}) = [\pi_{R,t}^{j}(-1),\pi_{P,t}^{j}(1),\pi_{S,t}^{j}(0)]
\end{align*}
$$

where $\pi_{A,t}^{j}$ is the adversary $j$'s probability of playing action $A$ (adversary policy) at time $t$ and the number in parentheses is the outcome if this action is chosen. This is a Game Theory notation for a lottery. In this case, each time the learner chooses an action, she is actually choosing a lottery over the outcomes $\{-1,0,1\}$.

#### Learner's Objective and Policy

The objective of the learner is to maximize her total reward $S_{n} = \sum_{t=1}^{n}X_{t}$.

Given the bandit problem, the learner needs a strategy to make the best decisions, with the aim to maximize her total reward. Since the learner is constrained by a finite number of interactions with the environment, she has to balance exploration with exploitation. Exploration is needed to gather data over actions, with the objective of finding the best action (or set of best actions), while exploitation is required to extract the maximum amount of reward from the environment, by choosing the best (inferred) action.

The strategy is set by a policy "function $\pi : ([k]\times [0,1])^{*} \to \mathcal{P_{k-1}}$, mapping history sequences to distributions over actions (regardless of measurability)." [1] Here the reward space is defined to be $[0,1]$. Thus, in a dynamic environment, given a history of actions and rewards, the learner updates her policy at each time $t$, following some maximizing reward algorithm. 

We can evaluate the learner's policy by using the regret, that can be defined in different ways. In the stochastic setting, the learner's policy is evaluated relative to the best policy $\pi^{*}$, whereas in the adversarial case, we compare the learner's actions with the best actions in hindsight.

$$
\begin{align*}
	R_{n}(\pi) = n \max_{i \in [k]} \mu_{i} - \mathbb{E}_{\pi}\left[ \sum_{t=1}^{n}x_{A_{t},t} \right] \qquad \text{(Stochastic Regret)}
\end{align*}
$$

$$
\begin{align*}
	R_{n}(\pi) = \max_{i \in [k]}\sum_{t=1}^{n}x_{i,t} - \mathbb{E}_{\pi}\left[ \sum_{t=1}^{n}x_{A_{t},t} \right] \qquad \text{(Adversarial Regret)} 
\end{align*}
$$

### Exp3 Algorithm

The Exp3 (Exponential-weight algorithm for exploration and exploitation) is an adversarial bandit algorithm. Consider a policy $\pi$ such that, for action $i$ at time $t$,

$$ \pi_{i,t} = \mathbb{P}(A_t = i | A_1,X_1,\dots,A_{t-1},X_{t-1})$$

where $\pi_{i,t}>0$, for all $i$ and $t$, and $X_{t}$ is the reward at time $t$. Consider that the learner only observes the reward of the action sampled from her policy. The importance-weighted estimator of $x_{i,t}$, the true reward of action $i$ at time $t$, is

$$
\begin{equation}
\hat{X}_{i,t} = \frac{\mathbb{I}_{\{A_{t} = i\}} X_{t}}{\pi_{i,t}},
\end{equation}
$$

This is an unbiased estimate of $x_{i,t}$ conditioned on the history observed after $t-1$ rounds. Indeed,

$$
\begin{align*}
	\mathbb{E}\left[\hat{X}_{i,t}|A_1,X_1,\dots,A_{t-1},X_{t-1}\right] &= \mathbb{E}_{t}\left[\hat{X}_{i,t}\right] \\
	&= \mathbb{E}_{t}\left[\frac{\mathbb{I}_{\{A_{t} = i\}} X_{t}}{\pi_{i,t}}\right]\\
	&= \mathbb{E}_{t}\left[\mathbb{I}_{\{A_{t} = i\}}\right] \frac{\mathbb{E}[X_{t}]}{\pi_{i,t}}\\
	&= x_{i,t}
\end{align*}
$$

Despite being an unbiased estimator, it may still possess high variance. Its variance is $$\mathbb{V}[\hat{X}_{i,t}] = \frac{x^{2}_{i,t}(1-\pi_{i,t})}{\pi_{i,t}}$$, that can be extremely large when $\pi_{i,t}$ is small and $x_{i,t}$ is bounded away from zero [1]. An alternative unbiased estimator is

$$
\begin{equation}
	\hat{X}_{i,t} = 1 - \frac{\mathbb{I}_{\left\{A_{t}=i \right\}}}{\pi_{i,t}}(1- X_{t})
\end{equation}
$$
 
Now consider $$\hat{S}_{i,t} = \sum_{s=1}^{t}\hat{X}_{i,t}$$, the total estimated reward for action $i$ by the end of round $t$. We can use this estimate to update the policy $\pi_{i,t}$, by using the following update rule:

$$
\begin{equation*}
	\pi_{i,t} = \frac{ \exp\left(\eta\hat{S}_{i,t-1}\right)}{\sum_{j=1}^{k}\exp\left(\eta\hat{S}_{j,t-1}\right)}
\end{equation*}
$$

where $\eta$ is the exploitation rate: when is high, it favors exploitation, while when is low it favors exploration. This hyperparameter can be fixed or vary through time, thus depending on the number of arms and/or the horizon. (Actually, in [1] and elsewhere, $\eta$ is called the learning rate, but I think calling it exploitation rate makes more sense).

The EXP3 algorithm is summarized as follows:

>`Exp3 Algorithm`\
> Input: $n,k,\eta$\
> Set $$\hat{S}_{0,i} = 0$$ for all $$i \in [k]$$\
> for $t=1$ to $n$ do\
> 	$$\qquad \pi_{t,i} = \frac{\exp(\eta\hat{S}_{t-1,i})}{\sum_{j=1}^{k}\exp(\eta\hat{S}_{t-i,j})}$$, for all $$i \in [k]$$\
> 	$\qquad  A_{t} \sim \pi_{t}$ and observe $X_{t}$\
> 	$$\qquad \hat{S}_{t,i} = \hat{S}_{t-1,i} + 1 - \frac{\mathbb{I}_{\left\{A_{t}=i \right\}}(1- X_{t})}{\pi_{t,i}}$$ $$\left[\text{ or } \hat{S}_{t,i} = \hat{S}_{t-1,i} + \frac{\mathbb{I}_{\left\{A_{t}=i \right\}}X_{t}}{\pi_{t,i}}\right]$$
> 
----------


### Simulation

This section presents some simple simulations, with no intention of being a rigorous analysis. The goal is to show how the Exp3 algorithm works and how it can be used in the RPS game. 

#### Stochastic Setting

Using the Exp3 Algorithm, we simulate a RPS game with one adversary that has a fixed policy. The game is played over $300$ rounds, the adversary's policy is $\pi^{0} = [\pi_{\tiny R}^{0}, \pi_{\tiny P}^{0}, \pi_{\tiny S}^{0}] = [0.55 ,0.40 , 0.05]$ and the learner's exploitation rate is $0.4$. The rewards are multiplied by $0.1$ to scale them down. The following figure shows the path for the learner's policy over the $300$ rounds.

![Learner's Policy Path](/assets/img/policy_path.svg){:style="display:block; margin-left:auto; margin-right:auto"}

To use the regret to measure how well this policy performs, we define the following regret functions:

$$
\begin{equation*}
	\hat{R}_{n} = n \mu^{*} - \sum_{t=1}^{n} X_{t} \qquad \text{(random regret)}
\end{equation*}
$$

$$
\begin{equation*}
	\bar{R}_{n} = n \mu^{*} -  \sum_{t=1}^{n} \mu_{A_{t}} \qquad \text{(pseudo-regret)}
\end{equation*}
$$

where $$\mu^{*}$$ is the mean reward of the best action. We will use the pseudo-regret to measure the performance of the Exp3 algorithm. For our example we have $\mu = [\mu_{R}, \mu_{P}, \mu_{S}] = [-0.035, 0.05, -0.015] \implies \mu^{*}= 0.05$.

![Policy Path - Regret](/assets/img/regret_chart.svg){:style="display:block; margin-left:auto; margin-right:auto"}

We can see that the regret is decreasing until it stabilizes around $2.5$. This is expected, since the learner is playing against a fixed policy, and the optimal action is obvious: play paper.

Another example is where the adversary's fixed policy is more balanced, like $\pi^{0} = [0.35 ,0.40 , 0.25]$. In this case, the mean for each action is: $\mu_{R} = -0.01$, $\mu_{P} = 0.015$, $\mu_{S} = -0.005 \implies \mu^{*} = \mu_{P}$. Different from the previous example, playing paper as the optimal action is not obvious, a fact that reveals itself when we simulate the game, where the learner takes time to find the optimal policy. The figure below shows the performance of a learner in this environment, through $300$ rounds with an exploitation rate of $0.4$. We can see that after $300$ rounds the learner is still not playing the optimal action.

![Policy Path - Regret](/assets/img/path_regret_2.svg){:style="display:block; margin-left:auto; margin-right:auto"}

Note that the Exp3 algorithm does not take into account the rewards of the actions not played to update the policy, only using the reward realized. This is inefficient since the learner is not using all available information to maximize her total return.

#### Adversarial Setting 

In the adversarial setting, we have a real game where the adversary responds to the learner's actions, adapting its own policy following some objective, like maximizing its total rewards. In the figure below, the learner plays with an adversary that uses the Exp3 algorithm to update its policy, in the same way as the learner. The game is played over $1000$ rounds, and the players' exploitation rate is $0.1$.

![Policy Path - Adversarial](/assets/img/adversarial_1.svg){:style="display:block; margin-left:auto; margin-right:auto"}

We can see that the adversary starts with a high probability of playing Rock, and the leaner responds by increasing its probability of playing Paper, followed by the adversary increasing its probability of playing Scissors, and so on. Because of the high variance of the Exp3 algorithm, the trajectories of the policies are not smooth and an equilibrium is highly unlikely, but the overall trend is clear.

#### Conclusion

In conclusion, bandit algorithms such as the Exp3 algorithm are useful tools for solving problems where we must make decisions under uncertainty. By balancing the exploration and exploitation of different options, these algorithms can learn which actions yield the highest rewards. As demonstrated in the rock, paper, scissors game, the Exp3 algorithm is able to learn and adapt to its opponent's behavior, allowing it to achieve a decent performance. Overall, bandit algorithms offer a good approach for optimizing decision-making in a some applications.

#### Resources

In calculating the exponentials of the Exp3 algorithm, I encountered some numerical instability. To stabilize it, I used the approach presented [here](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/). In addition, the book Bandit Algorithms was used as the main reference, from which I obtained the algorithm framework and some of the definitions.

[1] Lattimore, T., & Szepesvári, C. (2020). Bandit Algorithms. Cambridge: Cambridge University Press. doi:10.1017/9781108571401\
[2] https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/\
[3] [code repository](https://github.com/fe-lipe-c/ML_Games_Group/tree/master/Rock_Paper_Scissors) 
