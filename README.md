# Name: Swaraj Bhanja | Student Id: st125052

# Welcome to Masked Language Model!

This is a web-based end-to-end application named Masked Language Model. It leverages the power of deep learning and web development to provide a website that performs textual inference based on the input premise and hypothesis sentences.

# About the Deep Learning Model

The brains of this solution is the deep learning model trained for this purpose. The DL model was trained based on the HuggingFace **Book Corpus** dataset. The complete **Book Corpus** dataset is a large collection of English-language books, primarily used in natural language processing (NLP) tasks. It contains the text of over 11000 books, which are predominantly fiction, offering a rich and diverse array of narratives and writing styles. The dataset was originally created to train large-scale language models, such as those used in text generation, language understanding, and transfer learning tasks. When loading only 2% of the dataset, as in the provided code snippet, it allows for quick experimentation and testing while retaining a representative sample of the dataset's language complexity and variety.

# Vocabularization

A custom vocabulary is created for the text data and converts the text into numerical tokens that can be used as input for machine learning models. First, it generates a list of unique words (word_list) from the cleaned text by joining all sentences into a single string, splitting by whitespace, and using `set()` to remove duplicates. The word2id dictionary is then initialized with special tokens such as `[PAD]`, `[CLS]`, `[SEP]`, and `[MASK]`, which are often used in natural language processing tasks for padding, classification, sentence separation, and masking, respectively. The main vocabulary is populated using a loop with tqdm which  maps each unique word to a unique ID. The `id2word` dictionary is created as a reverse mapping of word2id, allowing quick lookups from IDs back to words when needed. The `vocab_size` variable stores the total size of the vocabulary, including the special tokens.

# Tokenization

The custom vocabulary text data is converted into a list of tokens `(token_list)`. For each sentence, it splits the sentence into individual words and replaces each word with its corresponding ID from `word2id`. This results in a list of lists, where each inner list contains the tokenized (numerical) representation of a sentence. This transformation is crucial because machine learning models generally work with numerical data rather than raw text. The end result is a fully prepared dataset of tokenized sentences, ready for input into an NLP model.

# Preparing Data Using Batch Loader

A function called `make_batch()` is defined that generates a batch of training data for a machine learning model, specifically designed for tasks like language modeling or next-sentence prediction, commonly used in transformer-based models like BERT. The goal is to create a balanced batch of sentence pairs, where half of the pairs are "positive" (i.e., the second sentence logically follows the first) and the other half are "negative" (i.e., the sentences are unrelated or randomly paired). The function starts by initializing an empty batch and counters for positive and negative samples. It then repeatedly selects random sentence pairs from the preprocessed `token_list`, transforming them into input sequences by adding special tokens `[CLS]` at the start and `[SEP]` between and after sentences. It also generates `segment_ids`, a list distinguishing between tokens of the first and second sentences, which helps the model differentiate between them during training.

For the masked language modeling (MLM) task, the function randomly selects 15% of tokens from the input sequence to be masked, ensuring at least one token is masked but not exceeding a predefined `max_mask` limit. These selected positions are stored in `masked_pos`, and the original tokens are saved in `masked_tokens`. When masking, 80% of the time the token is replaced with `[MASK]`, 10% of the time it is replaced with a random token, and 10% of the time it remains unchanged. This randomness is crucial for training robust language models.

To ensure uniform input size, the function pads `input_ids` and `segment_ids` with zeros up to `max_len`, and similarly pads `masked_tokens` and `masked_pos` to max_mask. The function then determines whether the sentence pair is a valid "IsNext" pair (i.e., the second sentence immediately follows the first in the original text) and appends the processed data to the batch with an appropriate label (True for positive pairs and False for negative ones). This process continues until the batch contains an equal number of positive and negative samples, ensuring a balanced and effective training set. Finally, the function returns the completed batch, which is ready to be fed into the model for training.

# About the various classes

A custom implementation of the BERT (Bidirectional Encoder Representations from Transformers) model is made using PyTorch, tailored for tasks like masked language modeling and next sentence prediction. The BERT class inherits from `nn.Module`, making it compatible with PyTorch's neural network framework. During initialization `(__init__ method)`, the model accepts several hyperparameters, including the number of layers `(n_layers)`, attention heads (n_heads)`, model dimensions `(d_model)`, feed-forward dimensions `(d_ff)`, attention key dimensions `(d_k)`, the number of possible segment types `(n_segments)`, vocabulary size `(vocab_size)`, and the maximum sequence length `(max_len)`. These parameters are stored in a dictionary for easy access.

The model architecture consists of an embedding layer, a stack of encoder layers, and multiple linear layers for classification tasks. The embedding layer `(self.embedding)` converts input tokens into dense vectors, incorporating positional and segment embeddings, which help the model understand the order of tokens and distinguish between different segments (e.g., sentence pairs). The encoder layers `(self.layers)` are created using `nn.ModuleList`, each utilizing multi-head self-attention and feed-forward neural networks to capture contextual relationships between tokens. The `self.fc` and `self.classifier` layers are used for the next sentence prediction (NSP) task, where the model decides if the second sentence logically follows the first. The decoder layer shares weights with the embedding layer to reduce the total number of parameters, facilitating efficient training and better generalization. Additionally, a bias parameter `(self.decoder_bias)` is introduced for the masked language modeling (MLM) task.

The forward method handles the input processing during model training. First, the embedding layer generates the input representation, and then the output passes through the encoder layers, applying attention masks `(enc_self_attn_mask)` to ignore padding tokens during attention calculations. The NSP task is addressed by focusing on the `[CLS]` token at the start of each input, transforming it with a fully connected layer and the tanh activation function before classification. For the MLM task, the model gathers the representations of masked positions `(masked_pos)` and normalizes them using layer normalization `(self.norm)` and the GELU activation function `(F.gelu)`. The decoder then predicts the original masked tokens by outputting logits over the entire vocabulary.

Additionally, the `get_last_hidden_state` method allows the retrieval of the final hidden states from the last encoder layer, which can be useful for feature extraction or further fine-tuning tasks. Overall, this implementation of BERT is designed to handle both pre-training tasks (MLM and NSP) while offering flexibility for downstream applications in natural language understanding and generation.

# Training (BERT)

The training loop of the BERT model is executed, where it processes input data, computes losses for masked language modeling (MLM) and next sentence prediction (NSP) tasks, and updates the model's parameters to minimize these losses. Initially, the input data `(input_ids, segment_ids, masked_tokens, masked_pos, and isNext)` is prepared by converting each element of the batch into PyTorch tensors using torch.LongTensor. The `map` and `zip(*batch)` functions efficiently unpack the batch and apply the tensor conversion in one step. Subsequently, all inputs are moved to the GPU (device) to leverage faster computations.

The training loop is wrapped with `tqdm` to display a progress bar, which is especially useful for monitoring the training progress when dealing with long epochs. At the start of each epoch, `optimizer.zero_grad()` clears any accumulated gradients from the previous step, ensuring they do not interfere with the current optimization step. The model is then called with the prepared inputs, generating two outputs: `logits_lm` for the MLM task and `logits_nsp` for the NSP task. These outputs represent the predicted logits, or unnormalized probabilities, for each token in the vocabulary and for the binary classification of sentence continuity, respectively.

The MLM loss `(loss_lm)` is computed using the criterion, typically a cross-entropy loss function. The `logits_lm` tensor is transposed to match the expected dimensions of `(batch_size, vocab_size, max_mask)`, which allows for comparison with masked_tokens, representing the true masked words. The `.mean()` method computes the average loss across the batch, stabilizing the training process. The NSP loss `(loss_nsp)` is similarly calculated by comparing `logits_nsp` with `isNext`, which holds binary labels indicating whether the second sentence follows the first.

The total loss `(loss)` is the sum of loss_lm and loss_nsp, balancing both pre-training objectives of BERT. During training, every 100 epochs, the current loss is displayed, providing insight into how well the model is learning. The backpropagation step `(loss.backward())` computes gradients of the loss concerning all model parameters, and `optimizer.step()` updates the parameters in the direction that minimizes the loss. This iterative process of forward and backward passes continues for the specified number of epochs (num_epoch), gradually improving the model's ability to predict masked tokens and correctly classify sentence pairs.

# Training (S-BERT)

A Siamese Network built is on top of \the above BERT model for a sentence pair classification task, such as natural language inference (NLI). A Siamese Network involves two identical neural networks (in this case, BERT model) that process two inputs separately but share the same weights. The architecture is particularly useful when comparing two inputs, such as determining the semantic similarity between two sentences.

The training loop is set to run for `num_epoch` epochs, where the model and the additional `classifier_head` are set to training mode using `model.train()` and `classifier_head.train()`. The data loader `(train_dataloader)` is wrapped in `tqdm` to display a progress bar during training. At the start of each batch, the gradients are reset to avoid accumulation using `optimizer.zero_grad()` and `optimizer_classifier.zero_grad()`. The input data, including token IDs, attention masks, and segment IDs for both premise `(inputs_ids_a, attention_a, segment_ids_a)` and hypothesis `(inputs_ids_b, attention_b, segment_ids_b)`, are loaded onto the GPU `(device)`. The labels `(label)` are also transferred to the GPU for loss calculation.

The Siamese Network part involves passing both sentence inputs independently through the BERT model's `get_last_hidden_state()` method. This method extracts the final hidden states from the BERT model, which are then mean-pooled using a helper function `(mean_pool)` to obtain sentence embeddings `(u_mean_pool and v_mean_pool)`. These embeddings represent the semantic meaning of the input sentences as fixed-length vectors.

To capture the relationship between the two sentence embeddings, the absolute difference between the embeddings is computed `(uv_abs = torch.abs(uv))`, where uv is the element-wise difference between `u_mean_pool` and `v_mean_pool`. These vectors are then concatenated `(torch.cat)` to form a combined input (x) for the classifier head. The concatenated vector includes the original embeddings and their absolute difference, allowing the model to learn both direct information from each sentence and the comparative relationship between them.

The classifier head `(classifier_head)` processes this concatenated tensor to produce logits representing the predicted class probabilities. The softmax loss `(criterion)` is then computed by comparing these predictions (x) with the true labels (label). The backpropagation step `(loss.backward())` calculates the gradients, and the optimizers `(optimizer and optimizer_classifier)` update the model weights to minimize the loss. The learning rate schedulers `(scheduler and scheduler_classifier)` are also updated at each step to dynamically adjust the learning rate, which can improve training stability and convergence speed.

At the end of each epoch, the training loss is logged, providing insight into the model's learning progress. The Siamese Network architecture enhances the model's ability to understand and classify relationships between paired inputs, making it a powerful choice for sentence similarity and semantic matching tasks in NLP.

## Testing (S-BERT)

A prediction function for a natural language inference (NLI) task using a BERT-based Siamese Network with an additional cosine similarity measure is defined to assess the semantic closeness of two sentences. The function `predict_nli_class_with_similarity` takes a premise and a hypothesis as input, along with the model, classifier head, tokenizer, and device (e.g., GPU or CPU). The goal is to determine whether the hypothesis entails, contradicts, or is neutral concerning the premise, a classic NLI classification problem.

The process begins with tokenizing the input sentences using the provided tokenizer, which converts the sentences into input IDs and attention masks that the BERT model can understand. The segment IDs are initialized as zero tensors, indicating that both inputs belong to the same input type, which is essential for consistent model behavior. The function then computes BERT embeddings for both the premise and the hypothesis using the `model.get_last_hidden_state method`. The `torch.no_grad()` context is used to prevent gradient computation, optimizing memory usage and speed during inference.

Next, the mean-pooling method is applied to obtain fixed-length embeddings for each sentence, reducing the last hidden states to meaningful sentence vectors. These vectors are then used to construct the feature vector for classification by concatenating the embeddings of both sentences along with their absolute difference `(|u-v|)`. The combined tensor (x) is passed through the classifier head, which generates logits representing the predicted class probabilities.

The softmax function `(F.softmax)` is applied to convert logits into probability distributions over the three possible NLI classes: 'contradiction', 'neutral', and 'entailment'. The predicted class is determined by identifying the index of the maximum probability using `torch.argmax`. Additionally, the function calculates the cosine similarity between the mean-pooled embeddings of the premise and hypothesis using scikit-learn's `cosine_similarity` function. The cosine similarity provides a numerical measure of how semantically similar the two sentences are, with 1.0 indicating identical vectors and 0.0 indicating orthogonal (unrelated) vectors.

[Analysis Metrics and Challenges Discussion]([https://github.com/st125052/a3-nlp-machine-translation-language-st125052/blob/main/notebooks/pdfs/Attentions%20Analysis%20in%20Tasks%202%20and%203.pdf](https://github.com/st125052/a4-nlp-masked-language-model-st125052/blob/main/notebooks/pdfs/Analysis%20Metrics%20and%20Challenges%20Discussion.pdf))

## Pickling The Model
The S-BERT model was chosen for deployment.
> The pickled model was saved using a .pkl extension to be used later in a web-based application

# Website Creation
The model was then hosted over the Internet with Flask as the backend, HTML, CSS, JS as the front end, and Docker as the container. The end-user is presented with a UI wherein a search input box is present. Once the user types in the first set of words, they click on the `Infer Natural Language` button. The input texts iare sent to the JS handler which makes an API call to the Flask backend. The Flask backend has the GET route which intercepts the HTTP request. The input text is then fed to the model to generate the textual similarity. However, this happens via another API call to a Colab VM where the model is hosted. This translation is called tunneling and can be done via Ngrok. The result is then returned back to the JS handler as a list by the Flask backend. The JS handler then appends each token in the received list into the result container's inner HTML and finally makes it visible for the output to be shown. 

A Vanilla architecture was chosen due to time constraints. In a more professional scenario, the ideal approach would be used frameworks like React, Angular and Vue for Frontend and ASP.NET with Flask or Django for Backend.

The following describes the key points of the hosting discussion.
> **1. DigitalOcean (Hosting Provider)**
> 
>> - **Role:** Hosting and Server Management
>> - **Droplet:** Hosts the website on a virtual server, where all files, databases, and applications reside.
>> - **Dockerized Container:** The website is hosted in a Dockerized container running on the droplet. The container is built over a Ubuntu Linux 24.10 image.
>> - **Ports and Flask App:** The Dockerized container is configured to host the website on port 8000. It forwards requests to port 5000, where the Flask app serves the backend and static files. This flask app contains the pickled model, which is used for prediction.
>> - **IP Address:** The droplet’s public IP address directs traffic to the server.
>
>  **In Summary:** DigitalOcean is responsible for hosting the website within a Dockerized container, ensuring it is online and accessible via its IP address.
> 
>  **2. GoDaddy (Domain Registrar)**
>
>> - **Role:** Domain Registration and Management
>> - **Domain Purchase:** Registers and manages the domain name.
>> - **DNS Management:** Initially provided DNS setup, allowing the domain to be pointed to the DigitalOcean droplet’s IP address.
> 
> **In Summary:** GoDaddy ensures the domain name is registered and correctly points to the website’s hosting server.
>
>  **3. Cloudflare (DNS and Security/Performance Optimization)**
>
>> - **Role:** DNS Management, Security, and Performance Optimization
>> - **DNS Management:** Resolves the domain to the correct IP address, directing traffic to the DigitalOcean droplet.
>> - **CDN and Security:** Caches website content globally, enhances performance, and provides security features like DDoS protection and SSL encryption.
> 
> **In Summary:** Cloudflare improves the website’s speed, security, and reliability.
>
> **How It Works Together:**
> 
>> - **Domain Resolution:** The domain is registered with GoDaddy, which points it to Cloudflare's DNS servers. Cloudflare resolves the domain to the DigitalOcean droplet's IP address.
>> - **Content Delivery:** Cloudflare may serve cached content or forward requests to DigitalOcean, which processes and serves the website content to users.
> 
> **Advantages of This Setup:**
>
>> - **Security:** Cloudflare provides DDoS protection, SSL/TLS encryption, and a web application firewall.
>> - **Performance:** Cloudflare’s CDN reduces load times by caching content globally, while DigitalOcean offers scalable hosting resources.
>> - **Reliability:** The combination of GoDaddy, Cloudflare, and DigitalOcean ensures the website is always accessible, with optimized DNS resolution and robust hosting.



# Demo
https://github.com/user-attachments/assets/8c501a05-ed80-44ca-a3f1-472ad543ba19


# Access The Final Website
You can access the website [here](https://aitmltask.online). 

# Limitations
Note that currently the model is hosted on an ephemeral Colab VM, which disconnects upon inactivity. In a production scenario, this can be mitigated by using the HuggingFace Inference API or a VM that has built in Nvidia GPU support.
Also, the model predicts incorrect outputs for many pairs of premise and hypothesis, which can be attributed to the fact that the model was trained on a very small portion of the actual dataset.


# How to Run the Masked Language Model Docker Container Locally
### Step 1: Clone the Repository
> - First, clone the repository to your local machine.
### Step 2: Install Docker
> - If you don't have Docker installed, you can download and install it from the [Docker](https://www.docker.com) website.
### Step 3: Build and Run the Docker Container
Once Docker is installed, navigate to the app folder in the project directory. Delete the docker-compose-deployment.yml file and run the following commands to build and run the Docker container:
> - `docker compose up -d`

### Important Notes
> - The above commands will serve the Docker container on port 5000 and forward the requests to the Flask application running on port 5000 in the containerized environment.
> - Ensure Ports Are Free: Make sure that port 5000 is not already in use on your machine before running the container.
> - Changing Flask's Port: If you wish to change the port Flask runs on (currently set to 5000), you must update the port in the app.py file. After making the change, remember to rebuild the Docker image in the next step. Execute the following command to stop the process: `docker compose down`. Then goto Docker Desktop and delete the container and image from docker. 
