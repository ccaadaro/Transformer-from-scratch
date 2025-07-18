import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import random

# ===== Dataset =====
class DummyTranslationDataset(Dataset):
    def __init__(self, vocab_size=1000, seq_len=10, num_samples=1000):
        self.data = [(torch.randint(2, vocab_size, (seq_len,)), torch.randint(2, vocab_size, (seq_len,))) for _ in range(num_samples)]
        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        return src, tgt

# ===== Positional Encoding =====
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# ===== Attention Mechanisms =====
def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        # Expand mask to match scores dimensions
        if mask.dim() == 4:  # (B, 1, 1, S) or (B, 1, T, T)
            mask = mask.expand(scores.size(0), scores.size(1), scores.size(2), scores.size(3))
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, V), attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        B, T_q, _ = Q.shape
        _, T_k, _ = K.shape
        Q = self.W_q(Q).view(B, T_q, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(B, T_k, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(B, T_k, self.num_heads, self.d_k).transpose(1, 2)
        out, _ = scaled_dot_product_attention(Q, K, V, mask)
        out = out.transpose(1, 2).contiguous().view(B, T_q, -1)
        return self.fc_out(out)

# ===== Feedforward Layer =====
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

# ===== Encoder & Decoder Layers =====
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn))
        ff = self.ff(x)
        return self.norm2(x + self.dropout2(ff))

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        x = self.norm1(x + self.dropout1(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout2(self.cross_attn(x, enc_out, enc_out, src_mask)))
        return self.norm3(x + self.dropout3(self.ff(x)))

# ===== Transformer Model =====
class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=256, num_heads=4, num_layers=2, d_ff=512, dropout=0.1, max_len=100):
        super().__init__()
        self.src_embed = nn.Embedding(src_vocab, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, tgt_vocab)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src = self.pos_enc(self.src_embed(src))
        tgt = self.pos_enc(self.tgt_embed(tgt))
        for layer in self.encoder_layers:
            src = layer(src, src_mask)
        for layer in self.decoder_layers:
            tgt = layer(tgt, src, src_mask, tgt_mask)
        return self.fc_out(tgt)

# ===== Training Utilities =====
def create_mask(src, tgt):
    src_mask = (src != 0).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S)
    tgt_pad_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
    tgt_len = tgt.size(1)
    tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len))).bool().to(tgt.device)
    tgt_mask = tgt_pad_mask & tgt_sub_mask
    return src_mask, tgt_mask

# ===== Training Loop =====
def train():
    dataset = DummyTranslationDataset()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = Transformer(1000, 1000).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(5):
        model.train()
        total_loss = 0
        for src, tgt in dataloader:
            src, tgt = src.cuda(), tgt.cuda()
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            src_mask, tgt_mask = create_mask(src, tgt_input)
            output = model(src, tgt_input, src_mask, tgt_mask)
            output = output.view(-1, output.size(-1))
            tgt_output = tgt_output.contiguous().view(-1)
            loss = criterion(output, tgt_output)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}")
        # Inference test
        model.eval()
        src, _ = dataset[0]
        src = src.unsqueeze(0).cuda()
        src_mask, _ = create_mask(src, src)
        output = greedy_decode(model, src, src_mask)
        print(f"Source: {src.squeeze().tolist()}")
        print(f"Predicted: {output.squeeze().tolist()}")

def greedy_decode(model, src, src_mask, max_len=10, start_symbol=1):
    memory = src
    ys = torch.ones(src.size(0), 1).fill_(start_symbol).type_as(src)  # start token
    for i in range(max_len - 1):
        tgt_mask = create_mask(src, ys)[1]
        out = model(src, ys, src_mask, tgt_mask)
        prob = out[:, -1, :]
        next_word = torch.argmax(prob, dim=-1).unsqueeze(1)
        ys = torch.cat([ys, next_word], dim=1)
    return ys


# ===== Real Use Cases =====

def create_simple_vocab():
    """Create a simple vocabulary for more interpretable tests"""
    vocab = {
        '<pad>': 0, '<start>': 1, '<end>': 2,
        'hello': 3, 'world': 4, 'good': 5, 'morning': 6,
        'how': 7, 'are': 8, 'you': 9, 'fine': 10,
        'thanks': 11, 'bye': 12, 'see': 13, 'later': 14,
        'hola': 15, 'mundo': 16, 'buenos': 17, 'días': 18,
        'cómo': 19, 'estás': 20, 'bien': 21, 'gracias': 22,
        'adiós': 23, 'hasta': 24, 'luego': 25
    }
    return vocab

def text_to_tokens(text, vocab):
    """Convert text to tokens using vocabulary"""
    tokens = [vocab.get(word.lower(), vocab['<pad>']) for word in text.split()]
    return torch.tensor(tokens, dtype=torch.long)

def tokens_to_text(tokens, vocab):
    """Convert tokens to text using inverted vocabulary"""
    inv_vocab = {v: k for k, v in vocab.items()}
    return ' '.join([inv_vocab.get(token.item(), '<unk>') for token in tokens if token.item() != 0])

class SimpleTranslationDataset(Dataset):
    """Simple dataset for English-Spanish translation"""
    def __init__(self):
        self.vocab = create_simple_vocab()
        self.data = [
            ("hello world", "hola mundo"),
            ("good morning", "buenos días"),
            ("how are you", "cómo estás"),
            ("fine thanks", "bien gracias"),
            ("see you later", "hasta luego"),
            ("bye world", "adiós mundo"),
            ("hello how are you", "hola cómo estás"),
            ("good morning world", "buenos días mundo"),
        ]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        src_text, tgt_text = self.data[idx]
        src_tokens = text_to_tokens(src_text, self.vocab)
        tgt_tokens = torch.cat([
            torch.tensor([self.vocab['<start>']]),
            text_to_tokens(tgt_text, self.vocab),
            torch.tensor([self.vocab['<end>']])
        ])
        
        # Padding to fixed length
        max_len = 10
        src_padded = torch.zeros(max_len, dtype=torch.long)
        tgt_padded = torch.zeros(max_len, dtype=torch.long)
        
        src_padded[:min(len(src_tokens), max_len)] = src_tokens[:max_len]
        tgt_padded[:min(len(tgt_tokens), max_len)] = tgt_tokens[:max_len]
        
        return src_padded, tgt_padded

def test_translation_model():
    """Test the model with a simple translation dataset"""
    print("=== English-Spanish Translation Test ===")
    
    dataset = SimpleTranslationDataset()
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    vocab = dataset.vocab
    vocab_size = len(vocab)
    
    # Smaller model for testing
    model = Transformer(vocab_size, vocab_size, d_model=128, num_heads=4, 
                       num_layers=2, d_ff=256, dropout=0.1).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # Training
    print("Training translation model...")
    for epoch in range(20):
        model.train()
        total_loss = 0
        for src, tgt in dataloader:
            src, tgt = src.cuda(), tgt.cuda()
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            src_mask, tgt_mask = create_mask(src, tgt_input)
            output = model(src, tgt_input, src_mask, tgt_mask)
            output = output.view(-1, output.size(-1))
            tgt_output = tgt_output.contiguous().view(-1)
            
            loss = criterion(output, tgt_output)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}")
    
    # Translation tests
    print("\n=== Translation Results ===")
    model.eval()
    test_sentences = [
        "hello world",
        "good morning",
        "how are you",
        "see you later"
    ]
    
    with torch.no_grad():
        for sentence in test_sentences:
            src_tokens = text_to_tokens(sentence, vocab)
            src_padded = torch.zeros(10, dtype=torch.long)
            src_padded[:min(len(src_tokens), 10)] = src_tokens[:10]
            src = src_padded.unsqueeze(0).cuda()
            
            src_mask, _ = create_mask(src, src)
            output = greedy_decode_with_vocab(model, src, src_mask, vocab)
            
            predicted_text = tokens_to_text(output.squeeze(), vocab)
            print(f"English: {sentence}")
            print(f"Prediction: {predicted_text}")
            print("-" * 30)

def greedy_decode_with_vocab(model, src, src_mask, vocab, max_len=10):
    """Greedy decoding with specific vocabulary"""
    start_symbol = vocab['<start>']
    end_symbol = vocab['<end>']
    
    ys = torch.ones(src.size(0), 1).fill_(start_symbol).type_as(src)
    
    for i in range(max_len - 1):
        tgt_mask = create_mask(src, ys)[1]
        out = model(src, ys, src_mask, tgt_mask)
        prob = out[:, -1, :]
        next_word = torch.argmax(prob, dim=-1).unsqueeze(1)
        ys = torch.cat([ys, next_word], dim=1)
        
        # Stop if end token is generated
        if next_word.item() == end_symbol:
            break
    
    return ys

def test_attention_visualization():
    """Test for attention pattern visualization"""
    print("\n=== Attention Visualization Test ===")
    
    # Create a simple sequence
    vocab = create_simple_vocab()
    sentence = "hello world"
    src_tokens = text_to_tokens(sentence, vocab)
    src_padded = torch.zeros(10, dtype=torch.long)
    src_padded[:len(src_tokens)] = src_tokens
    src = src_padded.unsqueeze(0).cuda()
    
    # Simple model
    model = Transformer(len(vocab), len(vocab), d_model=64, num_heads=2, 
                       num_layers=1, d_ff=128).cuda()
    
    model.eval()
    with torch.no_grad():
        src_mask, _ = create_mask(src, src)
        
        # Forward pass to get representations
        embedded = model.pos_enc(model.src_embed(src))
        print(f"Input sequence: {tokens_to_text(src.squeeze(), vocab)}")
        print(f"Embedding shape: {embedded.shape}")
        print(f"First 3 dimensions of first token embedding: {embedded[0, 0, :3]}")

def test_model_capacity():
    """Test model capacity with different sizes"""
    print("\n=== Model Capacity Test ===")
    
    configs = [
        {"d_model": 64, "num_heads": 2, "num_layers": 1, "d_ff": 128},
        {"d_model": 128, "num_heads": 4, "num_layers": 2, "d_ff": 256},
        {"d_model": 256, "num_heads": 8, "num_layers": 3, "d_ff": 512},
    ]
    
    vocab = create_simple_vocab()
    vocab_size = len(vocab)
    
    for i, config in enumerate(configs):
        print(f"\nConfiguration {i+1}: {config}")
        model = Transformer(vocab_size, vocab_size, **config).cuda()
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Speed test
        src = torch.randint(1, vocab_size, (4, 10)).cuda()
        tgt = torch.randint(1, vocab_size, (4, 10)).cuda()
        
        import time
        start_time = time.time()
        with torch.no_grad():
            for _ in range(100):
                output = model(src, tgt)
        end_time = time.time()
        
        print(f"Time for 100 forward passes: {(end_time - start_time)*1000:.2f} ms")

def run_all_tests():
    """Execute all transformer tests"""
    print("🚀 Starting Transformer Tests\n")
    
    # Test 1: Basic training with random data
    print("1️⃣ Basic training with random data")
    train()
    
    # Test 2: Translation with real vocabulary
    print("\n2️⃣ Translation test with real vocabulary")
    test_translation_model()
    
    # Test 3: Attention visualization
    print("\n3️⃣ Attention visualization test")
    test_attention_visualization()
    
    # Test 4: Model capacity analysis
    print("\n4️⃣ Model capacity analysis")
    test_model_capacity()
    
    print("\n✅ All tests completed!")

if __name__ == "__main__":
    run_all_tests()
