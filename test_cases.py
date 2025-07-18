#!/usr/bin/env python3
"""
Specific use cases for the Transformer
"""

import torch
import torch.nn as nn
from transformer import Transformer, create_mask
import matplotlib.pyplot as plt
import numpy as np

def test_sequence_copying():
    """Test the model's ability to copy sequences"""
    print("=== Sequence Copying Test ===")
    
    # Create training data for copying sequences
    vocab_size = 20
    seq_len = 8
    num_samples = 1000
    
    # Generate data: output equals input but with special tokens
    train_data = []
    for _ in range(num_samples):
        seq = torch.randint(2, vocab_size, (seq_len,))
        # Add start and end tokens
        src = seq
        tgt = torch.cat([torch.tensor([1]), seq, torch.tensor([0])])  # <start> + seq + <end>
        train_data.append((src, tgt))
    
    # Model
    model = Transformer(vocab_size, vocab_size, d_model=128, num_heads=4, 
                       num_layers=2, d_ff=256, dropout=0.1)
    if torch.cuda.is_available():
        model = model.cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # Training
    print("Training copying model...")
    for epoch in range(50):
        total_loss = 0
        for i in range(0, len(train_data), 32):  # batch_size = 32
            batch = train_data[i:i+32]
            src_batch = torch.stack([item[0] for item in batch])
            tgt_batch = torch.stack([item[1] for item in batch])
            
            if torch.cuda.is_available():
                src_batch, tgt_batch = src_batch.cuda(), tgt_batch.cuda()
            
            tgt_input = tgt_batch[:, :-1]
            tgt_output = tgt_batch[:, 1:]
            
            src_mask, tgt_mask = create_mask(src_batch, tgt_input)
            output = model(src_batch, tgt_input, src_mask, tgt_mask)
            output = output.view(-1, output.size(-1))
            tgt_output = tgt_output.contiguous().view(-1)
            
            loss = criterion(output, tgt_output)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss / (len(train_data)//32):.4f}")
    
    # Test
    print("\n=== Copying Results ===")
    model.eval()
    test_cases = [
        torch.tensor([2, 5, 8, 3, 12, 7, 4, 9]),
        torch.tensor([15, 3, 7, 11, 2, 18, 6, 14]),
        torch.tensor([4, 4, 4, 4, 4, 4, 4, 4]),
    ]
    
    with torch.no_grad():
        for i, test_seq in enumerate(test_cases):
            if torch.cuda.is_available():
                test_seq = test_seq.cuda()
            
            src = test_seq.unsqueeze(0)
            src_mask, _ = create_mask(src, src)
            
            # Greedy decoding starting with start token
            ys = torch.ones(src.size(0), 1).fill_(1).type_as(src)  # <start>
            for j in range(seq_len):
                tgt_mask = create_mask(src, ys)[1]
                out = model(src, ys, src_mask, tgt_mask)
                prob = out[:, -1, :]
                next_word = torch.argmax(prob, dim=-1).unsqueeze(1)
                ys = torch.cat([ys, next_word], dim=1)
                
                # Stop if end token is generated
                if next_word.item() == 0:
                    break
            
            print(f"Case {i+1}:")
            print(f"  Input:  {test_seq.tolist()}")
            predicted = ys.squeeze()[1:].tolist()
            # Remove end token if exists
            if 0 in predicted:
                predicted = predicted[:predicted.index(0)]
            print(f"  Output: {predicted}")
            
            # Calculate accuracy
            min_len = min(len(test_seq), len(predicted))
            if min_len > 0:
                accuracy = sum(1 for a, b in zip(test_seq[:min_len].tolist(), predicted[:min_len]) if a == b) / min_len
                print(f"  Accuracy: {accuracy:.2%}")
            else:
                print(f"  Accuracy: 0.00%")
            print()

def test_arithmetic_sequences():
    """Test learning arithmetic sequences"""
    print("=== Arithmetic Sequences Test ===")
    
    # Generate simple arithmetic sequences
    def generate_arithmetic_sequence(start, diff, length):
        return [start + i * diff for i in range(length)]
    
    # Create training data
    vocab_size = 100
    seq_len = 6
    num_samples = 2000
    
    train_data = []
    for _ in range(num_samples):
        start = torch.randint(1, 10, (1,)).item()  # Reduce range
        diff = torch.randint(1, 4, (1,)).item()   # Reduce range
        sequence = generate_arithmetic_sequence(start, diff, seq_len)
        
        # Ensure all values are within vocabulary range
        if max(sequence) < vocab_size:
            # Convert to tensors
            src = torch.tensor(sequence[:seq_len//2])  # First elements
            tgt_full = torch.tensor(sequence[seq_len//2:])  # Elements to predict
            # Add special tokens
            tgt = torch.cat([torch.tensor([1]), tgt_full, torch.tensor([0])])
            train_data.append((src, tgt))
    
    print(f"Generated {len(train_data)} training examples")
    
    # Model
    model = Transformer(vocab_size, vocab_size, d_model=128, num_heads=4, 
                       num_layers=3, d_ff=256, dropout=0.1)
    if torch.cuda.is_available():
        model = model.cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # Training
    print("Training arithmetic sequences model...")
    for epoch in range(100):
        total_loss = 0
        for i in range(0, len(train_data), 32):
            batch = train_data[i:i+32]
            src_batch = torch.stack([item[0] for item in batch])
            tgt_batch = torch.stack([item[1] for item in batch])
            
            if torch.cuda.is_available():
                src_batch, tgt_batch = src_batch.cuda(), tgt_batch.cuda()
            
            tgt_input = tgt_batch[:, :-1]
            tgt_output = tgt_batch[:, 1:]
            
            src_mask, tgt_mask = create_mask(src_batch, tgt_input)
            output = model(src_batch, tgt_input, src_mask, tgt_mask)
            output = output.view(-1, output.size(-1))
            tgt_output = tgt_output.contiguous().view(-1)
            
            loss = criterion(output, tgt_output)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss / (len(train_data)//32):.4f}")
    
    # Test
    print("\n=== Arithmetic Sequences Results ===")
    model.eval()
    test_cases = [
        ([2, 4, 6], [8, 10, 12]),      # +2
        ([1, 3, 5], [7, 9, 11]),       # +2
        ([3, 6, 9], [12, 15, 18]),     # +3
        ([2, 5, 8], [11, 14, 17]),     # +3
    ]
    
    with torch.no_grad():
        for i, (input_seq, expected) in enumerate(test_cases):
            src = torch.tensor(input_seq).unsqueeze(0)
            if torch.cuda.is_available():
                src = src.cuda()
            
            src_mask, _ = create_mask(src, src)
            
            # Greedy decoding starting with start token
            ys = torch.ones(src.size(0), 1).fill_(1).type_as(src)
            predicted_nums = []
            
            for j in range(len(expected) + 1):  # +1 in case of end token
                tgt_mask = create_mask(src, ys)[1]
                out = model(src, ys, src_mask, tgt_mask)
                prob = out[:, -1, :]
                next_word = torch.argmax(prob, dim=-1).unsqueeze(1)
                ys = torch.cat([ys, next_word], dim=1)
                
                # Stop if end token is generated
                if next_word.item() == 0:
                    break
                elif next_word.item() != 1:  # Don't include start token
                    predicted_nums.append(next_word.item())
            
            print(f"Case {i+1}:")
            print(f"  Input:     {input_seq}")
            print(f"  Expected:  {expected}")
            print(f"  Predicted: {predicted_nums}")
            
            # Calculate element-wise accuracy
            if len(predicted_nums) >= len(expected):
                accuracy = sum(1 for a, b in zip(expected, predicted_nums[:len(expected)]) if a == b) / len(expected)
                print(f"  Accuracy: {accuracy:.2%}")
            else:
                print(f"  Accuracy: 0.00% (incomplete sequence)")
            print()

def test_memory_capacity():
    """Test the model's memory capacity"""
    print("=== Memory Capacity Test ===")
    
    # Create sequences with memory patterns
    vocab_size = 50
    seq_lengths = [5, 10, 15, 20]
    
    for seq_len in seq_lengths:
        print(f"\nTesting with sequences of length {seq_len}...")
        
        # Generate data: remember the first element
        train_data = []
        for _ in range(1000):
            seq = torch.randint(2, vocab_size, (seq_len,))
            # Output is the first element with special tokens
            target = torch.cat([torch.tensor([1]), seq[0:1], torch.tensor([0])])
            train_data.append((seq, target))
        
        # Model with more capacity for long sequences
        model = Transformer(vocab_size, vocab_size, d_model=256, num_heads=8, 
                           num_layers=4, d_ff=512, dropout=0.1, max_len=seq_len+10)
        if torch.cuda.is_available():
            model = model.cuda()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        # Longer training for difficult sequences
        epochs = 50 if seq_len <= 10 else 100
        for epoch in range(epochs):
            total_loss = 0
            for i in range(0, len(train_data), 16):  # smaller batch size
                batch = train_data[i:i+16]
                src_batch = torch.stack([item[0] for item in batch])
                tgt_batch = torch.stack([item[1] for item in batch])
                
                if torch.cuda.is_available():
                    src_batch, tgt_batch = src_batch.cuda(), tgt_batch.cuda()
                
                tgt_input = tgt_batch[:, :-1]
                tgt_output = tgt_batch[:, 1:]
                
                src_mask, tgt_mask = create_mask(src_batch, tgt_input)
                output = model(src_batch, tgt_input, src_mask, tgt_mask)
                output = output.view(-1, output.size(-1))
                tgt_output = tgt_output.contiguous().view(-1)
                
                loss = criterion(output, tgt_output)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        
        # Evaluation
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for i in range(100):  # Test 100 cases
                seq = torch.randint(2, vocab_size, (seq_len,))
                target = seq[0].item()
                
                if torch.cuda.is_available():
                    seq = seq.cuda()
                
                src = seq.unsqueeze(0)
                src_mask, _ = create_mask(src, src)
                
                # Greedy decoding
                ys = torch.ones(src.size(0), 1).fill_(1).type_as(src)
                tgt_mask = create_mask(src, ys)[1]
                out = model(src, ys, src_mask, tgt_mask)
                prob = out[:, -1, :]
                prediction = torch.argmax(prob, dim=-1)
                
                if prediction.item() == target:
                    correct += 1
                total += 1
        
        accuracy = correct / total
        print(f"  Accuracy: {accuracy:.2%}")
        print(f"  (Remember first element of {seq_len} elements)")

def test_sequence_reversal():
    """Test the model's ability to reverse sequences"""
    print("=== Sequence Reversal Test ===")
    
    # Create training data for reversing sequences
    vocab_size = 30
    seq_len = 6
    num_samples = 2000
    
    train_data = []
    for _ in range(num_samples):
        seq = torch.randint(2, vocab_size, (seq_len,))
        # Output is the reversed sequence
        reversed_seq = torch.flip(seq, [0])
        # Add special tokens
        tgt = torch.cat([torch.tensor([1]), reversed_seq, torch.tensor([0])])
        train_data.append((seq, tgt))
    
    # Model
    model = Transformer(vocab_size, vocab_size, d_model=128, num_heads=4, 
                       num_layers=3, d_ff=256, dropout=0.1)
    if torch.cuda.is_available():
        model = model.cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # Training
    print("Training reversal model...")
    for epoch in range(80):
        total_loss = 0
        for i in range(0, len(train_data), 32):
            batch = train_data[i:i+32]
            src_batch = torch.stack([item[0] for item in batch])
            tgt_batch = torch.stack([item[1] for item in batch])
            
            if torch.cuda.is_available():
                src_batch, tgt_batch = src_batch.cuda(), tgt_batch.cuda()
            
            tgt_input = tgt_batch[:, :-1]
            tgt_output = tgt_batch[:, 1:]
            
            src_mask, tgt_mask = create_mask(src_batch, tgt_input)
            output = model(src_batch, tgt_input, src_mask, tgt_mask)
            output = output.view(-1, output.size(-1))
            tgt_output = tgt_output.contiguous().view(-1)
            
            loss = criterion(output, tgt_output)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss / (len(train_data)//32):.4f}")
    
    # Test
    print("\n=== Reversal Results ===")
    model.eval()
    test_cases = [
        torch.tensor([2, 5, 8, 3, 12, 7]),
        torch.tensor([10, 4, 6, 9, 2, 15]),
        torch.tensor([3, 3, 5, 5, 7, 7]),
        torch.tensor([20, 15, 10, 5, 25, 18]),
    ]
    
    with torch.no_grad():
        for i, test_seq in enumerate(test_cases):
            if torch.cuda.is_available():
                test_seq = test_seq.cuda()
            
            src = test_seq.unsqueeze(0)
            src_mask, _ = create_mask(src, src)
            expected = torch.flip(test_seq, [0]).tolist()
            
            # Greedy decoding
            ys = torch.ones(src.size(0), 1).fill_(1).type_as(src)
            predicted_nums = []
            
            for j in range(seq_len + 1):
                tgt_mask = create_mask(src, ys)[1]
                out = model(src, ys, src_mask, tgt_mask)
                prob = out[:, -1, :]
                next_word = torch.argmax(prob, dim=-1).unsqueeze(1)
                ys = torch.cat([ys, next_word], dim=1)
                
                if next_word.item() == 0:  # End token
                    break
                elif next_word.item() != 1:  # Don't include start token
                    predicted_nums.append(next_word.item())
            
            print(f"Case {i+1}:")
            print(f"  Input:     {test_seq.tolist()}")
            print(f"  Expected:  {expected}")
            print(f"  Predicted: {predicted_nums}")
            
            # Calculate accuracy
            if len(predicted_nums) >= len(expected):
                accuracy = sum(1 for a, b in zip(expected, predicted_nums[:len(expected)]) if a == b) / len(expected)
                print(f"  Accuracy: {accuracy:.2%}")
            else:
                print(f"  Accuracy: 0.00% (incomplete sequence)")
            print()

if __name__ == "__main__":
    print("🧪 Running Transformer Specific Use Cases\n")
    
    test_sequence_copying()
    print("\n" + "="*50 + "\n")
    
    test_arithmetic_sequences()
    print("\n" + "="*50 + "\n")
    
    test_memory_capacity()
    print("\n" + "="*50 + "\n")
    
    test_sequence_reversal()
    
    print("\n✅ All specific tests completed!")
