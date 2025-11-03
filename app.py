import streamlit as st
import torch
import re
import torch.nn as nn
import torch.nn.functional as F
import os

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Silicon GPU (MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using NVIDIA GPU (CUDA)")
else:
    device = torch.device("cpu")
    print("Using CPU")
def prepare_data(window_size):
  
    # read file
    text = open('dataset-1.txt', 'r', encoding='utf-8').read()
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # text = re.sub(r'(\n\s*\n)+', ' . . . . . ', text) # replace multiple newlines with ' . . . . . '
    text = re.sub(r'[-]+', ' ', text) # replace hyphens with space
    text = re.sub(r'[^a-zA-Z0-9 \.\n]', '', text) # remove special characters except period
    text = re.sub(r'\n\n', ' <EOS> ', text) # replace newlines with space
    text = re.sub(r'\.+', '.', text).strip() # collapse multiple spaces into one
    text = re.sub(r'\.', ' . ', text) # surround periods with spaces
    text = re.sub(r'\s+', ' ', text).strip() # collapse multiple spaces into one
    text = text.lower() # convert to lowercase
    paragraphs = text.split(' <eos> ')
    # words = [word for para in paragraphs for word in para.split() + ['<EOS>']]
    words = []
    for para in paragraphs:
        words.extend(para.split(' ') + ['<PAD>'] * window_size)
    X, Y = [], []
    for i in range(len(words)):
        start_idx = max(0, i - window_size)
        context = words[start_idx:i]
        context = ['<PAD>'] * (window_size - len(context)) + context  # left padding
        target = words[i]

        X.append(context)
        Y.append(target)

    from collections import Counter

    word_counts = Counter(words)
    vocab = sorted(word_counts.keys())
    vocab_size = len(vocab)

    most_common_10 = word_counts.most_common(10)
    least_common_10 = word_counts.most_common()[-10:]

    stoi = {word: i for i, word in enumerate(vocab)}
    itos = {i: word for word, i in stoi.items()}

    return stoi, itos





class Nextword(nn.Module):
    def __init__(self, block_size, vocab_size, emb_dim=64, hidden_size=1024, activation='tanh'):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.activation = activation
        self.fc1 = nn.Linear(block_size * emb_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        # x: (batch, block_size)
        if self.activation == 'relu':
            
            x = self.emb(x)                           # (batch, block_size, emb_dim)
            x = x.view(x.shape[0], -1)                # flatten
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)                           # logits
        else:
            x = self.emb(x)                           # (batch, block_size, emb_dim)
            x = x.view(x.shape[0], -1)                # flatten
            x = F.tanh(self.fc1(x))
            x = F.tanh(self.fc2(x))
            x = self.fc3(x)                           # logits
        return x



def generate_text(model, stoi, itos, block_size, device, start_context=None, max_len=20, temperature=1.0):
    """
    Generate a sequence of words from a trained model.

    Args:
        model: Trained PyTorch language model
        stoi: dict, mapping from word → index
        itos: dict, mapping from index → word
        block_size: int, context size expected by the model
        device: torch device ('cuda' or 'cpu')
        start_context: list of str (optional), seed words
        max_len: int, number of words to generate

    Returns:
        str: Generated text sequence
    """

    model.eval()  # evaluation mode (no dropout, etc.)

    # --- Initialize context ---
    if start_context is None:
        context = [stoi['<PAD>']] * block_size  # start with padding
    else:
        
        context = [stoi.get(w, stoi['<PAD>']) for w in start_context]
        context = context[-block_size:]
        context = [stoi['<PAD>']] * (block_size - len(context)) + context
        generated_words = []

    # --- Generate words one by one ---
    with torch.no_grad():
        for _ in range(max_len):
            x = torch.tensor(context).view(1, -1).to(device)
            y_pred = model(x)  # logits for next word
            # adding temperature
            y_pred = y_pred / temperature
            ix = torch.distributions.categorical.Categorical(logits=y_pred).sample().item()
            word = itos[ix]

            generated_words.append(word)

            # slide the context window forward
            context = context[1:] + [ix]

    model.train()  # restore training mode

    i = 0
    new_words = []
    while i < len(generated_words):
        if i + 2 < len(generated_words) and generated_words[i:i+3] == ['<PAD>', '<PAD>', '<PAD>']:
            while i < len(generated_words) and generated_words[i] == '<PAD>':
                i += 1
            new_words.append('\n')
        else:
            if generated_words[i] != '<PAD>':
                new_words.append(generated_words[i])
            i += 1
    return ' '.join(new_words)


#file path for model
file_path = 'my_model'
models = {}
folder_path = "mymodels"  # your folder name

# List all files in the folder
files = os.listdir(folder_path)

# Print all model files (e.g. .pt or .pth)
model_files = [f for f in files if f.endswith(".pth")]
model_params = {}
for model_file in model_files:
    para = model_file.split('_')
    model_name = model_file.split('.')[0]
    block_size = int(para[1])
    emb_dim = int(para[2])
    activation = para[3]
    seed = para[4].split('.')[0]
    stoi = prepare_data(block_size)[0]
    itos = prepare_data(block_size)[1]
    temp_model = Nextword(block_size, len(stoi), emb_dim, activation= activation).to(device)
    temp_model = torch.compile(temp_model)
    try:
        temp_model.load_state_dict(torch.load(os.path.join(folder_path, model_file), map_location=torch.device(device), weights_only=True))
    except Exception as e:
        print(f"Error loading model {model_file}: {e}")
        continue
    models[model_name] = temp_model
    model_params[model_name] = (block_size, emb_dim, activation, seed)

    
print(model_params)
#development streamlit app 

# taking input from user
st.title("Next Word Prediction Model")
st.write("This app predicts the next word based on the input context using pre-trained models.")

st.divider()

st.subheader("Model Selection")
model_display_names = {
    model_name: f"Block: {params[0]} | Emb: {params[1]} | Activation: {params[2].upper()} | Seed: {params[3]}"
    for model_name, params in model_params.items()
}

selected_display = st.selectbox(
    "Choose a model:",
    list(model_display_names.values()),
    help="Select based on context window, embedding size, activation function, and seed"
)

# Get the actual model name from the display name
model_choice = [k for k, v in model_display_names.items() if v == selected_display][0]

if model_choice:
    block_size, emb_dim, activation, seed = model_params[model_choice]
    st.write("**Model Parameters:**")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Block Size", block_size)
    with col2:
        st.metric("Embedding Dim", emb_dim)
    with col3:
        st.metric("Activation", activation.title())
    with col4:
        st.metric("Seed", seed.title())
    st.divider()
context_input = st.text_input("Enter context words (separated by spaces):")
# adding temperature slider
temperature = st.slider("Select temperature for text generation", min_value=0.1, max_value=3.0, value=1.0, step=0.1)
max_length = st.slider("Select maximum length of generated text", min_value=5, max_value=50, value=20)
if st.button("Generate Text"):

    block_size, emb_dim, activation, seed = model_params[model_choice]
    stoi, itos = prepare_data(block_size)
    context_words = context_input.strip().split(' ')
    generated_text = generate_text(models[model_choice], stoi, itos, block_size, device, start_context=context_words, max_len=max_length,temperature=temperature)
    st.write("Generated Text:")
    st.write(generated_text)

