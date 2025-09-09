# patient_persona_fine_tuning.py
import json
import torch
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
import argparse

def create_synthetic_data():
    """Create synthetic doctor-patient conversation data with different personas"""
    synthetic_data = []
    
    # Define different patient personas
    personas = {
        "calm": "cooperative, measured, provides clear information",
        "anxious": "worried, rushed, seeks reassurance, uses emphatic language",
        "rude": "dismissive, impatient, questions competence, short responses",
        "overly_patient": "verbose, provides excessive detail, tangential storytelling"
    }
    
    # Sample conversation templates
    conversation_templates = [
        {
            "doctor": "Good morning, what brings you in today?",
            "patient_responses": {
                "calm": "Good morning, Doctor. I've had a persistent cough for about two weeks now.",
                "anxious": "Oh thank goodness you're here! I have this terrible cough and I'm really worried it might be something serious!",
                "rude": "Finally. I've been waiting forever. It's a cough. Can you just give me something for it?",
                "overly_patient": "Well, good morning to you too, Doctor. It's actually quite interesting how this all started. You see, about two weeks ago, maybe it was a Tuesday, no actually it was Wednesday because that's when I normally do my grocery shopping..."
            }
        },
        {
            "doctor": "Can you describe the pain on a scale of 1 to 10?",
            "patient_responses": {
                "calm": "I'd say it's about a 4. It's noticeable but doesn't prevent me from daily activities.",
                "anxious": "It's definitely a 9! Maybe even a 10! It's the worst pain I've ever felt!",
                "rude": "What kind of question is that? Just figure out what's wrong with me.",
                "overly_patient": "Well, that's an interesting question. On Monday it was probably a 3, but then on Tuesday after I had my morning tea - I drink Earl Grey, you know, with just a splash of milk - it went up to about a 5, and then yesterday..."
            }
        }
    ]
    
    # Generate training examples
    for conv in conversation_templates:
        for persona, description in personas.items():
            example = {
                "persona": persona,
                "description": description,
                "input": f"Doctor: {conv['doctor']}",
                "response": f"Patient: {conv['patient_responses'][persona]}"
            }
            synthetic_data.append(example)
    
    return synthetic_data

def format_instruction(example):
    """Format the training examples with system prompt"""
    system_prompt = f"You are a virtual patient. Generate a response that is {example['persona']} - {example['description']}."
    doctor_input = example['input'].replace('Doctor: ', '')
    patient_response = example['response'].replace('Patient: ', '')
    
    full_text = f"System: {system_prompt}\nDoctor: {doctor_input}\nPatient: {patient_response}"
    
    return {
        "text": full_text
    }

def fine_tune_model(data, model_name="microsoft/DialoGPT-small", output_dir="./patient_persona_model"):
    """Fine-tune a lightweight model on the patient persona data"""
    
    # Load tokenizer and model
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set padding token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        print("Trying GPT2 as fallback...")
        try:
            model_name = "gpt2"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
        except Exception as e2:
            print(f"Error loading GPT2: {e2}")
            return None, None
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")
    
    # Prepare dataset
    formatted_data = [format_instruction(d) for d in data]
    
    # Extract just the text for tokenization
    texts = [item["text"] for item in formatted_data]
    
    # Tokenize the entire dataset
    print("Tokenizing dataset...")
    tokenized_data = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    
    # Create labels (same as input_ids for causal LM)
    tokenized_data["labels"] = tokenized_data["input_ids"].clone()
    
    # Convert to dataset format
    dataset_dict = {
        "input_ids": tokenized_data["input_ids"],
        "attention_mask": tokenized_data["attention_mask"],
        "labels": tokenized_data["labels"]
    }
    
    dataset = Dataset.from_dict(dataset_dict)
    
    # Configure LoRA for efficient fine-tuning
    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["c_attn", "c_proj"]
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=1e-4,
        logging_steps=5,
        save_steps=50,
        eval_steps=50,
        save_total_limit=2,
        prediction_loss_only=True,
        remove_unused_columns=False,
        report_to=None,
        optim="adamw_torch",
        warmup_steps=20,
        lr_scheduler_type="linear",
        dataloader_pin_memory=False,
    )
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    
    # Start training
    print("Starting training...")
    trainer.train()
    
    # Save the model
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print("Training completed successfully!")
    return model, tokenizer

def generate_response(model, tokenizer, persona, description, doctor_input):
    """Generate a patient response given a persona and doctor input"""
    system_prompt = f"System: You are a virtual patient. Generate a response that is {persona} - {description}."
    input_text = f"Doctor: {doctor_input}\nPatient:"
    
    full_prompt = f"{system_prompt}\n{input_text}"
    
    # Move model to appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.8,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2
        )
    
    # Decode and return the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the patient response part
    if "Patient:" in generated_text:
        response = generated_text.split("Patient:")[-1].strip()
    else:
        response = generated_text.replace(full_prompt, "").strip()
    
    return response

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a model for patient persona generation")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--generate", action="store_true", help="Generate a response")
    parser.add_argument("--persona", type=str, help="Persona to use for generation")
    parser.add_argument("--input", type=str, help="Doctor's input for generation")
    args = parser.parse_args()
    
    # Create or load synthetic data
    data = create_synthetic_data()
    
    if args.train:
        print("Starting model training...")
        model, tokenizer = fine_tune_model(data)
        if model is None:
            print("Training failed. Please check your environment and try again.")
        else:
            print("Training completed successfully!")
    
    if args.generate:
        if not args.persona or not args.input:
            print("Please specify both --persona and --input for generation")
            return
        
        # Load the fine-tuned model
        try:
            tokenizer = AutoTokenizer.from_pretrained("./patient_persona_model")
            model = AutoModelForCausalLM.from_pretrained(
                "./patient_persona_model",
                torch_dtype=torch.float32
            )
        except Exception as e:
            print(f"Model not found or error loading: {e}")
            print("Please train the model first with --train")
            return
        
        # Find persona description
        persona_desc = next((d["description"] for d in data if d["persona"] == args.persona), "")
        
        # Generate response
        response = generate_response(model, tokenizer, args.persona, persona_desc, args.input)
        print(f"\nDoctor: {args.input}")
        print(f"Patient ({args.persona}): {response}")

if __name__ == "__main__":
    main()