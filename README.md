# AMD AI Premier League (AAIPL) Submission

This repository contains our submission for the AMD AI Premier League Hackathon. Our core strategy revolves around robust **inference-level optimization**, specifically through advanced **Prompt Tuning**, after initial tests suggested that our fine-tuning efforts were not yielding the expected performance gains.

This document outlines the different approaches we explored and provides instructions on how to run our final implementation.

---

## Strategic Approach: Inference Over Fine-Tuning

Our primary strategic decision was to focus our efforts on maximizing the performance of the base `Qwen3-4B` model at inference time. We believe that a well-designed, model-aware prompting strategy can outperform a hastily fine-tuned model, especially given the time and potential hardware constraints of a hackathon environment.

We explored three distinct paths to achieve this goal.

---

### Approach 1: Structured Programming with DSPy

Our initial plan was to leverage the **DSPy framework** to create structured, optimizable programs for both the Q-Agent and the A-Agent.

* **Concept:** DSPy abstracts away manual prompt engineering, allowing developers to define the data flow and let the framework optimize the underlying prompts. We designed programs with specific signatures and modules, intending to use a powerful model like GPT-4o for the optimization (teleprompting) step and then run the compiled, model-agnostic program using the local `Qwen3-4B` model for inference.
* **Outcome:** This approach was theoretically sound, as it separates the program's logic from the specific LLM. However, the true power of DSPy lies in its optimizers, which we found challenging to integrate seamlessly with a local model under the hackathon's conditions. This led us to explore more direct prompting methods.

---

### Approach 2: Deep-Dive into Prompt Tuning (Our Final Strategy) 🎯

This became our chosen strategy due to its directness, effectiveness, and lower operational overhead. Instead of relying on a framework to discover the best prompts, we took a research-driven approach to engineer them ourselves.

* **Methodology:**
    1.  **Technical Analysis:** We began by surveying the official **Qwen Technical Report** to understand the model's architecture, strengths, and inherent biases.
    2.  **Benchmark Research:** We reviewed several research papers that benchmarked `Qwen3-4B` and other leading Small Language Models (SLMs). This helped us identify common failure points and areas of struggle for models in this class.
    3.  **Domain Targeting:** Based on our research and the hackathon's domains (Logical Reasoning, Puzzles, etc.), we identified specific sub-domains and question archetypes where LLMs typically falter (e.g., complex spatial reasoning, multi-level negation, temporal puzzles).
* **Outcome:** We used these insights to craft highly-targeted "adversarial" prompts for our Q-Agent to challenge competitors, and robust, defensive prompts for our A-Agent to handle such tricky questions. This method proved to be the most reliable and performant within the competition's constraints.

---

### Approach 3: GAN-Inspired Adversarial Training

As a more experimental and ambitious strategy, we attempted to create a self-improving agent ecosystem inspired by Generative Adversarial Networks (GANs).

* **Concept:** We envisioned two DSPy programs (a Q-Agent "Generator" and an A-Agent "Discriminator") training simultaneously. The Q-Agent would learn to generate questions that fool the A-Agent, while the A-Agent would learn to better answer them. The loss from one agent would be used to strengthen the other, creating a competitive loop that would, in theory, result in highly sophisticated agents.
* **Outcome:** This proved to be the most challenging approach to implement. We encountered significant **GPU glitches** and found it difficult to reliably manage the adversarial loop using **LiteLLM with local models**. Due to these technical hurdles, we had to abandon this approach in favor of the more stable and directly controllable Prompt Tuning strategy.

---

## How to Run Our Solution 🚀

To run our final submission, please follow these simple steps:

1.  Take the `answer_agent.py` file from our submission.
2.  Take the `question_agent.py` file from our submission.
3.  Replace the corresponding files in the hackathon's base directory with our versions.

The scripts are designed to work out-of-the-box with the existing environment and will execute our Prompt Tuning strategy.
