#!/usr/bin/env python3
"""
Evaluation and metrics for the Transformer
"""

import torch
import torch.nn as nn
from transformer import Transformer, create_mask, SimpleTranslationDataset, tokens_to_text
import time
import json

def evaluate_model_performance():
    """Evaluate model performance with different metrics"""
    print("=== Model Performance Evaluation ===")
    
    # Model configurations to test
    configs = [
        {"name": "Small", "d_model": 64, "num_heads": 2, "num_layers": 1, "d_ff": 128},
        {"name": "Medium", "d_model": 128, "num_heads": 4, "num_layers": 2, "d_ff": 256},
        {"name": "Large", "d_model": 256, "num_heads": 8, "num_layers": 3, "d_ff": 512},
    ]
    
    results = {}
    
    for config in configs:
        print(f"\n--- Evaluating {config['name']} model ---")
        
        # Create model
        model_config = {k: v for k, v in config.items() if k != 'name'}
        model = Transformer(1000, 1000, **model_config)
        
        if torch.cuda.is_available():
            model = model.cuda()
        
        # Metrics
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Speed test
        batch_size = 32
        seq_len = 10
        src = torch.randint(1, 1000, (batch_size, seq_len))
        tgt = torch.randint(1, 1000, (batch_size, seq_len))
        
        if torch.cuda.is_available():
            src, tgt = src.cuda(), tgt.cuda()
        
        # Warmup
        model.eval()
        with torch.no_grad():
            for _ in range(10):
                _ = model(src, tgt)
        
        # Medir tiempo
        start_time = time.time()
        with torch.no_grad():
            for _ in range(100):
                _ = model(src, tgt)
        end_time = time.time()
        
        inference_time = (end_time - start_time) / 100
        
        # Medir uso de memoria
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
            
            with torch.no_grad():
                _ = model(src, tgt)
            
            peak_memory = torch.cuda.max_memory_allocated()
            memory_usage = (peak_memory - initial_memory) / 1024**2  # MB
        else:
            memory_usage = 0
        
        results[config['name']] = {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'inference_time': inference_time,
            'memory_usage': memory_usage,
            'config': model_config
        }
        
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Inference time: {inference_time*1000:.2f} ms")
        print(f"  Memory usage: {memory_usage:.2f} MB")
    
    return results

def benchmark_translation_quality():
    """Translation quality benchmark"""
    print("\n=== Translation Quality Benchmark ===")
    
    dataset = SimpleTranslationDataset()
    vocab = dataset.vocab
    vocab_size = len(vocab)
    
    # Train model
    model = Transformer(vocab_size, vocab_size, d_model=128, num_heads=4, 
                       num_layers=2, d_ff=256, dropout=0.1)
    if torch.cuda.is_available():
        model = model.cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    print("Training translation model...")
    
    # Training data
    train_data = [dataset[i] for i in range(len(dataset))]
    
    for epoch in range(50):
        model.train()
        total_loss = 0
        
        for i in range(0, len(train_data), 4):  # batch_size = 4
            batch = train_data[i:i+4]
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
            print(f"Epoch {epoch+1}, Loss: {total_loss / (len(train_data)//4):.4f}")
    
    # Evaluation
    print("\n--- Translation Evaluation ---")
    model.eval()
    
    # Test cases
    test_cases = [
        ("hello world", "hola mundo"),
        ("good morning", "buenos días"),
        ("how are you", "cómo estás"),
        ("fine thanks", "bien gracias"),
        ("see you later", "hasta luego"),
    ]
    
    correct_translations = 0
    total_translations = len(test_cases)
    
    with torch.no_grad():
        for src_text, expected_tgt in test_cases:
            # Convert to tokens
            src_tokens = torch.tensor([vocab.get(word.lower(), vocab['<pad>']) 
                                     for word in src_text.split()])
            src_padded = torch.zeros(10, dtype=torch.long)
            src_padded[:min(len(src_tokens), 10)] = src_tokens[:10]
            src = src_padded.unsqueeze(0)
            
            if torch.cuda.is_available():
                src = src.cuda()
            
            # Generate translation
            src_mask, _ = create_mask(src, src)
            start_symbol = vocab['<start>']
            end_symbol = vocab['<end>']
            
            ys = torch.ones(src.size(0), 1).fill_(start_symbol).type_as(src)
            
            for i in range(9):  # max_len - 1
                tgt_mask = create_mask(src, ys)[1]
                out = model(src, ys, src_mask, tgt_mask)
                prob = out[:, -1, :]
                next_word = torch.argmax(prob, dim=-1).unsqueeze(1)
                ys = torch.cat([ys, next_word], dim=1)
                
                if next_word.item() == end_symbol:
                    break
            
            # Convert to text
            predicted_text = tokens_to_text(ys.squeeze()[1:], vocab)
            predicted_text = predicted_text.replace('<end>', '').strip()
            
            print(f"Input: {src_text}")
            print(f"Expected: {expected_tgt}")
            print(f"Predicted: {predicted_text}")
            
            # Simple evaluation (exact match)
            if predicted_text == expected_tgt:
                correct_translations += 1
                print("✅ Correct")
            else:
                print("❌ Incorrect")
            print("-" * 30)
    
    accuracy = correct_translations / total_translations
    print(f"\nTranslation accuracy: {accuracy:.2%} ({correct_translations}/{total_translations})")
    
    return accuracy

def generate_performance_report():
    """Generate a complete performance report"""
    print("🏆 Generating Complete Performance Report")
    
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_performance": {},
        "translation_quality": 0,
        "system_info": {
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
            "pytorch_version": torch.__version__
        }
    }
    
    # Evaluate model performance
    print("\n1. Evaluating performance of different configurations...")
    report["model_performance"] = evaluate_model_performance()
    
    # Evaluate translation quality
    print("\n2. Evaluating translation quality...")
    report["translation_quality"] = benchmark_translation_quality()
    
    # Save report
    with open('performance_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print("\n📊 Report saved to 'performance_report.json'")
    
    # Show summary
    print("\n" + "="*60)
    print("REPORT SUMMARY")
    print("="*60)
    
    print(f"System: {report['system_info']['cuda_device']}")
    print(f"PyTorch: {report['system_info']['pytorch_version']}")
    print(f"CUDA available: {report['system_info']['cuda_available']}")
    print()
    
    print("Performance by configuration:")
    for name, metrics in report["model_performance"].items():
        print(f"  {name}:")
        print(f"    - Parameters: {metrics['total_params']:,}")
        print(f"    - Time: {metrics['inference_time']*1000:.2f} ms")
        print(f"    - Memory: {metrics['memory_usage']:.2f} MB")
    
    print(f"\nTranslation quality: {report['translation_quality']:.2%}")
    
    return report

if __name__ == "__main__":
    report = generate_performance_report()
    print("\n✅ Complete evaluation finished!")
