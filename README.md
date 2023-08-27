# Numbers every LLM Developer should know

[中文](https://github.com/NascentCore/llm-numbers-cn)

At Google, there was a document put together by [Jeff Dean](https://en.wikipedia.org/wiki/Jeff_Dean), the legendary engineer, called [Numbers every Engineer should know](http://brenocon.com/dean_perf.html). It’s really useful to have a similar set of numbers for LLM developers to know that are useful for back-of-the envelope calculations. Here we share particular numbers we at Anyscale use, why the number is important and how to use it to your advantage. 

## Notes on the Github version

Last updates: 2023-05-17

If you feel there's an issue with the accuracy of the numbers, please file an issue. Think there are more numbers that should be in this doc? Let us know or file a PR. 

We are thinking the next thing we should add here is some stats on tokens per second of different models. 

## Prompts


### 40-90%[^1]: Amount saved by appending “Be Concise” to your prompt

It’s important to remember that you pay by the token for responses. This means that asking an LLM to be concise can save you a lot of money. This can be broadened beyond simply appending “be concise” to your prompt: if you are using GPT-4 to come up with 10 alternatives, maybe ask it for 5 and keep the other half of the money. 


### 1.3:1 -- Average tokens per word

LLMs operate on tokens. Tokens are words or sub-parts of words, so “eating” might be broken into two tokens “eat” and “ing”. A 750 word document in English will be about 1000 tokens. For languages other than English, the tokens per word increases depending on their commonality in the LLM's embedding corpus.

Knowing this ratio is important because most billing is done in tokens, and the LLM’s context window size is also defined in tokens. 


## Prices[^2]

Prices are of course subject to change, but given how expensive LLMs are to operate, the numbers in this section are critical. We use OpenAI for the numbers here, but prices from other providers you should check out ([Anthropic](https://cdn2.assets-servd.host/anthropic-website/production/images/model_pricing_may2023.pdf), [Cohere](https://cohere.com/pricing)) are in the same ballpark. 


### ~50:1 -- Cost Ratio of GPT-4 to GPT-3.5 Turbo[^3] 

What this means is that for many practical applications, it’s much better to use GPT-4 for things like generating high quality fine tuning data, or for automated evaluation of other models -- things you might only do once instead of it living in the middle of your inference cycle. It is roughly 50 times cheaper to use GPT-3.5-Turbo than GPT-4 (the “roughly” is because GPT-4 charges differently for the prompt and the generated output)  – so you really need to check on how far you can get with GPT-3.5-Turbo. GPT-3.5-Turbo is more than enough for tasks like summarization for example. 


### 5:1 -- Cost Ratio of generation of text using GPT-3.5-Turbo vs OpenAI embedding 

This means it is way cheaper to look something up in a vector store than to ask an LLM to generate it. E.g. “What is the capital of Delaware?” when looked up in an neural information retrieval system costs about 5x[^4] less than if you asked GPT-3.5-Turbo. The cost difference compared to GPT-4 is a whopping 250x! 


### 10:1 -- Cost Ratio of OpenAI embedding to Self-Hosted embedding 

> Note: this number is sensitive to load and embedding batch size, so please consider this approximate. 

In our blog post, we noted that using a g4dn.4xlarge (on-demand price: $1.20/hr) we were able to embed at about 9000 tokens per second using Hugging Face’s SentenceTransformers (which are pretty much as good as OpenAI’s embeddings). Doing some basic math of that rate and that node type indicates it is considerably cheaper (factor of 10 cheaper) to self-host embeddings (and that is before you start to think about things like ingress and egress fees). 


### 6:1 -- Cost Ratio of OpenAI fine tuned vs base model queries

It costs you 6 times as much to serve a fine tuned model as it does the base model on OpenAI. This is pretty exorbitant, but might make sense because of the possible multi-tenancy of base models. It also means it is far more cost effective to tweak the prompt for a base model than to fine tune a customized model. 


### 1:1 -- Cost Ratio of Self-Hosted base vs fine-tuned model queries 

If you’re self hosting a model, then it more or less costs the same amount to serve a fine tuned model as it does to serve a base one: the models have the same number of parameters. 


## Training and Fine Tuning


### ~$1 million: Cost to train a 13 billion parameter model on 1.4 trillion tokens

The [LLaMa paper](https://arxiv.org/abs/2302.13971) mentions it took them 21 days to train LLaMa using 2048 GPUs A100 80GB GPUs. We considered training our own model on the Red Pajama training set, then we ran the numbers. The above is assuming everything goes right, nothing crashes, and the calculation succeeds on the first time, etc. Plus it involves the coordination of 2048 GPUs. That’s not something most companies can do (shameless plug time: of course, we at Anyscale can – that’s our [bread and butter](https://www.anyscale.com/blog/training-175b-parameter-language-models-at-1000-gpu-scale-with-alpa-and-ray)! Contact us if you’d like to learn more). The point is that training your own LLM is possible, but it’s not cheap. And it will literally take days to complete each run. Much cheaper to use a pre-trained model. 


### &lt; 0.001: Cost ratio of fine tuning vs training from scratch

This is a bit of a generalization, but the cost of fine tuning is negligible. We showed for example that you can fine tune a [6B parameter model for about $7](https://www.anyscale.com/blog/how-to-fine-tune-and-serve-llms-simply-quickly-and-cost-effectively-using). Even at OpenAI’s rate for its most expensive fine-tunable model, Davinci, it is 3c per 1000 tokens. That means to fine tune on the entire works of Shakespeare (about 1 million words), you’re looking at $40[^5]. However, fine tuning is one thing and training from scratch is another … 


## GPU Memory

If you’re self-hosting a model, it’s really important to understand GPU memory because LLMs push your GPU’s memory to the limit. The following statistics are specifically about inference. You need considerably more memory for training or fine tuning. 


### V100: 16GB, A10G: 24GB, A100: 40/80GB: GPU Memory Capacities

It may seem strange, but it’s important to know the amount of memory different types of GPUs have. This will cap the number of parameters your LLM can have. Generally, we like to use A10Gs because they cost $1.50 to $2 per hour each at AWS on-demand prices and have 24G of GPU memory, vs the A100s which will run you about $5 each at AWS on-demand prices. 


### 2x number of parameters: Typical GPU memory requirements of an LLM for serving

For example, if you have a 7 billion parameter model, it takes about 14GB of GPU space. This is because most of the time, one 16-bit float (or 2 bytes) is required per parameter. There’s usually no need to go beyond 16-bit accuracy, and most of the time when you go to 8-bit accuracy you start to lose resolution (though that may be acceptable in some cases). Of course there are efforts to reduce this, notably llama.cpp which runs a 13 billion parameter model on a 6GB GPU by quantizing aggressively down to 4 bits (and 8 bits without too much impact), but that’s atypical. 


### ~1GB: Typical GPU memory requirements of an embedding model

Whenever you are doing sentence embedding (a very typical thing you do for clustering, semantic search and classification tasks), you need an embedding model like [sentence transformers](https://www.sbert.net/docs/pretrained_models.html#sentence-embedding-models/). OpenAI also has its own embeddings that they provide commercially. 

You typically don’t have to worry about how much memory embeddings take on the GPU, they’re fairly small. We’ve even had the embedding and the LLM on the same GPU. 


### >10x: Throughput improvement from batching LLM requests 

Running an LLM query through a GPU is very high latency: it may take, say, 5 seconds, with a throughput of 0.2 queries per second.  The funny thing is, though, if you run two tasks, it might only take 5.2 seconds. This means that if you can bundle 25 queries together, it would take about 10 seconds, and our throughput has improved to 2.5 queries per second. However, see the next point. 


### ~1 MB: GPU Memory required for 1 token of output with a 13B parameter model

The amount of memory you need is directly proportional to the maximum number of tokens you want to generate. So for example, if you want to generate outputs of up to 512 tokens (about 380 words), you need 512MB. No big deal you might say – I have 24GB to spare, what’s 512MB? Well, if you want to run bigger batches it starts to add up. So if you want to do batches of 16, you need 8GB of space. There are some techniques being developed that overcome this, but it’s still a real issue. 

# Cheatsheet

<img width="1097" alt="Screenshot 2023-05-17 at 1 46 09 PM" src="https://github.com/ray-project/llm-numbers/assets/9677264/5d40c6a3-84d7-436a-8fc4-a8d58008765d">

# Next Steps

See our earlier [blog series on solving Generative AI infrastructure](https://www.anyscale.com/blog/ray-common-production-challenges-for-generative-ai-infrastructure) and [using LangChain with Ray](https://www.anyscale.com/blog/llm-open-source-search-engine-langchain-ray). \
 \
If you are interested in learning more about Ray, see [Ray.io](http://ray.io/) and [Docs.Ray.io](http://docs.ray.io/). \
 \
To connect with the Ray community join #LLM on the [Ray Slack](https://docs.google.com/forms/d/e/1FAIpQLSfAcoiLCHOguOm8e7Jnn-JJdZaCxPGjgVCvFijHB5PLaQLeig/viewform) or our [Discuss forum](https://discuss.ray.io/). \
 \
If you are interested in our Ray hosted service for ML Training and Serving, see [Anyscale.com/Platform ](http://www.anyscale.com/platform)and click the 'Try it now' button

**Ray Summit 2023:** If you are interested to learn much more about how Ray can be used to build performant and scalable LLM applications and fine-tune/train/serve LLMs on Ray, join [Ray Summit](https://raysummit.anyscale.com/) on September 18-20th! We have a set of great keynote speakers including John Schulman from OpenAI and Aidan Gomez from Cohere, community and tech talks about Ray as well as [practical training focused on LLMs](https://github.com/ray-project/ray-educational-materials/blob/main/NLP_workloads/Text_generation/LLM_finetuning_and_batch_inference.ipynb).

<!-- Footnotes themselves at the bottom. -->
## Notes

[^1]:
     Based on experimentation with GPT-3.5-Turbo using a suite of prompts on 2023-05-08. 

[^2]:
     Retrieved from [http://openai.com/pricing](http://openai.com/pricing) on 2023-05-08. 

[^3]:
      **GPT-4**: 6c/1k tokens for the prompt, 12c/1k tokens for the generation (32,000 window version, 8,000 window version is half that). **GPT-3.5 Turbo**: 0.2c/1k tokens. 

[^4]:
     This assumes the vector lookup is “free.” It’s not, but it uses CPUs (much cheaper) and is fairly fast. 

[^5]:
     1 million words / 0.75 tokens/word / 1000*0.03 = $40. 
