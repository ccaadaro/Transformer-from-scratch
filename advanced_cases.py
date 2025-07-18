#!/usr/bin/env python3
"""
Advanced use cases for the Transformer
"""

import torch
import torch.nn as nn
from transformer import Transformer, create_mask
import random
import numpy as np

def test_pattern_recognition():
    """Test complex pattern recognition"""
    print("=== Pattern Recognition Test ===")
    
    # Create specific patterns
    patterns = {
        'fibonacci': [1, 1, 2, 3, 5, 8, 13, 21],
        'squares': [1, 4, 9, 16, 25, 36, 49, 64],
        'primes': [2, 3, 5, 7, 11, 13, 17, 19],
        'powers_of_2': [1, 2, 4, 8, 16, 32, 64, 128],
    }
    
    vocab_size = 150
    
    # Generate training data
    train_data = []
    for pattern_name, pattern in patterns.items():
        for _ in range(500):  # 500 examples per pattern
            # Take a sliding window from the pattern
            start_idx = random.randint(0, len(pattern) - 6)
            input_seq = pattern[start_idx:start_idx+3]
            output_seq = pattern[start_idx+3:start_idx+6]
            
            # Convert to tensors
            src = torch.tensor(input_seq)
            tgt = torch.cat([torch.tensor([1]), torch.tensor(output_seq), torch.tensor([0])])
            train_data.append((src, tgt))
    
    # Model
    model = Transformer(vocab_size, vocab_size, d_model=256, num_heads=8, 
                       num_layers=4, d_ff=512, dropout=0.1)
    if torch.cuda.is_available():
        model = model.cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # Training
    print("Training pattern recognition model...")
    for epoch in range(100):
        total_loss = 0
        random.shuffle(train_data)  # Shuffle data
        
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
        
        if (epoch + 1) % 25 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss / (len(train_data)//32):.4f}")
    
    # Test
    print("\n=== Pattern Recognition Results ===")
    model.eval()
    test_cases = [
        ([1, 1, 2], [3, 5, 8], "Fibonacci"),
        ([4, 9, 16], [25, 36, 49], "Squares"),
        ([3, 5, 7], [11, 13, 17], "Primes"),
        ([2, 4, 8], [16, 32, 64], "Powers of 2"),
    ]
    
    with torch.no_grad():
        for i, (input_seq, expected, pattern_name) in enumerate(test_cases):
            src = torch.tensor(input_seq).unsqueeze(0)
            if torch.cuda.is_available():
                src = src.cuda()
            
            src_mask, _ = create_mask(src, src)
            
            # Greedy decoding
            ys = torch.ones(src.size(0), 1).fill_(1).type_as(src)
            predicted_nums = []
            
            for j in range(len(expected) + 1):
                tgt_mask = create_mask(src, ys)[1]
                out = model(src, ys, src_mask, tgt_mask)
                prob = out[:, -1, :]
                next_word = torch.argmax(prob, dim=-1).unsqueeze(1)
                ys = torch.cat([ys, next_word], dim=1)
                
                if next_word.item() == 0:
                    break
                elif next_word.item() != 1:
                    predicted_nums.append(next_word.item())
            
            print(f"Case {i+1} ({pattern_name}):")
            print(f"  Input:     {input_seq}")
            print(f"  Expected:  {expected}")
            print(f"  Predicted: {predicted_nums}")
            
            # Calculate accuracy
            if len(predicted_nums) >= len(expected):
                accuracy = sum(1 for a, b in zip(expected, predicted_nums[:len(expected)]) if a == b) / len(expected)
                print(f"  Accuracy:  {accuracy:.2%}")
            else:
                print(f"  Accuracy:  0.00% (incomplete sequence)")
            print()

def test_sorting_task():
    """Test sorting capability"""
    print("=== Sorting Test ===")
    
    vocab_size = 100
    seq_len = 5
    num_samples = 3000
    
    # Generate training data
    train_data = []
    for _ in range(num_samples):
        # Generate random sequence
        seq = torch.randint(2, vocab_size, (seq_len,))
        # Output is the sorted sequence
        sorted_seq = torch.sort(seq)[0]
        # Add special tokens
        tgt = torch.cat([torch.tensor([1]), sorted_seq, torch.tensor([0])])
        train_data.append((seq, tgt))
    
    # Model
    model = Transformer(vocab_size, vocab_size, d_model=256, num_heads=8, 
                       num_layers=4, d_ff=512, dropout=0.1)
    if torch.cuda.is_available():
        model = model.cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # Training
    print("Training sorting model...")
    for epoch in range(120):
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
        
        if (epoch + 1) % 30 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss / (len(train_data)//32):.4f}")
    
    # Test
    print("\n=== Sorting Results ===")
    model.eval()
    test_cases = [
        torch.tensor([5, 2, 8, 1, 6]),
        torch.tensor([9, 3, 7, 4, 2]),
        torch.tensor([15, 8, 12, 6, 10]),
        torch.tensor([3, 3, 1, 7, 1]),
    ]
    
    with torch.no_grad():
        for i, test_seq in enumerate(test_cases):
            if torch.cuda.is_available():
                test_seq = test_seq.cuda()
            
            src = test_seq.unsqueeze(0)
            src_mask, _ = create_mask(src, src)
            expected = torch.sort(test_seq)[0].tolist()
            
            # Greedy decoding
            ys = torch.ones(src.size(0), 1).fill_(1).type_as(src)
            predicted_nums = []
            
            for j in range(seq_len + 1):
                tgt_mask = create_mask(src, ys)[1]
                out = model(src, ys, src_mask, tgt_mask)
                prob = out[:, -1, :]
                next_word = torch.argmax(prob, dim=-1).unsqueeze(1)
                ys = torch.cat([ys, next_word], dim=1)
                
                if next_word.item() == 0:
                    break
                elif next_word.item() != 1:
                    predicted_nums.append(next_word.item())
            
            print(f"Case {i+1}:")
            print(f"  Input:     {test_seq.tolist()}")
            print(f"  Expected:  {expected}")
            print(f"  Predicted: {predicted_nums}")
            
            # Calculate accuracy
            if len(predicted_nums) >= len(expected):
                accuracy = sum(1 for a, b in zip(expected, predicted_nums[:len(expected)]) if a == b) / len(expected)
                print(f"  Accuracy:  {accuracy:.2%}")
            else:
                print(f"  Accuracy:  0.00% (incomplete sequence)")
            print()

def test_mathematical_operations():
    """Test simple mathematical operations"""
    print("=== Mathematical Operations Test ===")
    
    vocab_size = 200
    
    # Generate simple addition data
    train_data = []
    for _ in range(5000):
        a = random.randint(1, 50)
        b = random.randint(1, 50)
        result = a + b
        
        # Format: [a, 100, b] -> [result] (100 represents '+')
        src = torch.tensor([a, 100, b])
        tgt = torch.tensor([1, result, 0])  # <start> result <end>
        train_data.append((src, tgt))
    
    # Model
    model = Transformer(vocab_size, vocab_size, d_model=256, num_heads=8, 
                       num_layers=4, d_ff=512, dropout=0.1)
    if torch.cuda.is_available():
        model = model.cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # Training
    print("Training addition model...")
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
        
        if (epoch + 1) % 25 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss / (len(train_data)//32):.4f}")
    
    # Test
    print("\n=== Addition Results ===")
    model.eval()
    test_cases = [
        (15, 23, 38),
        (7, 12, 19),
        (25, 8, 33),
        (42, 17, 59),
    ]
    
    with torch.no_grad():
        for i, (a, b, expected) in enumerate(test_cases):
            src = torch.tensor([a, 100, b]).unsqueeze(0)
            if torch.cuda.is_available():
                src = src.cuda()
            
            src_mask, _ = create_mask(src, src)
            
            # Greedy decoding
            ys = torch.ones(src.size(0), 1).fill_(1).type_as(src)
            tgt_mask = create_mask(src, ys)[1]
            out = model(src, ys, src_mask, tgt_mask)
            prob = out[:, -1, :]
            prediction = torch.argmax(prob, dim=-1).item()
            
            print(f"Case {i+1}:")
            print(f"  Operation: {a} + {b}")
            print(f"  Expected:  {expected}")
            print(f"  Predicted: {prediction}")
            print(f"  Correct:   {prediction == expected}")
            print()

if __name__ == "__main__":
    print("🔬 Running Advanced Transformer Use Cases\n")
    
    test_pattern_recognition()
    print("\n" + "="*60 + "\n")
    
    test_sorting_task()
    print("\n" + "="*60 + "\n")
    
    test_mathematical_operations()
    
    print("\n🎯 All advanced tests completed!")
