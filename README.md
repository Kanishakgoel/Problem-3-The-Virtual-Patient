Patient Persona Voice Generation System
A lightweight AI system for generating realistic patient responses with different personalities, designed for medical training simulations in resource-constrained environments.

📋 Overview
This project fine-tunes a small language model to generate patient dialogue in various personas (calm, anxious, rude, overly patient) for medical training applications. The system is optimized to run on limited computational resources using parameter-efficient fine-tuning techniques.

🎯 Use Case
Medical Education in Rural Areas: This system enables medical students in rural India to practice patient interactions through VR training, overcoming the challenge of limited access to real patients.

✨ Features
Multiple Personas: Generates responses in 4 distinct patient personalities

Lightweight: Optimized for low-resource environments using LoRA and quantization

Easy Integration: Can be deployed with Ollama for local inference

Customizable: Easy to add new personas or training data

🏗️ Architecture
text
Synthetic Data → Fine-tuning → Lightweight Model → API/VR Integration
    │              (LoRA)         (Gemma-2B/         (Ollama/
    │                            DialoGPT-small)     FastAPI)
    │
    └── Persona Definitions
        ├── Calm
        ├── Anxious
        ├── Rude
        └── Overly Patient
🚀 Quick Start
Prerequisites
bash
pip install torch transformers datasets peft accelerate
Installation
Clone the repository:

bash
git clone https://github.com/your-username/patient-persona-generator.git
cd patient-persona-generator
Install dependencies:

bash
pip install -r requirements.txt
Training the Model
bash
# Train with default parameters
python patient_persona_fine_tuning.py --train

# Train with custom model
python patient_persona_fine_tuning.py --train --model_name "microsoft/DialoGPT-small"
Generating Responses
bash
# Generate a response with specific persona
python patient_persona_fine_tuning.py --generate --persona anxious --input "What symptoms are you experiencing?"

# Example outputs:
python patient_persona_fine_tuning.py --generate --persona calm --input "How can I help you today?"
python patient_persona_fine_tuning.py --generate --persona rude --input "Where does it hurt?"
python patient_persona_fine_tuning.py --generate --persona overly_patient --input "When did this start?"
🎭 Available Personas
Persona	Description	Example Response
Calm	Cooperative, measured, clear information	"Good morning, Doctor. I've had a persistent cough for about two weeks now."
Anxious	Worried, rushed, seeks reassurance	"Oh thank goodness you're here! I have this terrible cough and I'm really worried!"
Rude	Dismissive, impatient, questions competence	"Finally. I've been waiting forever. It's a cough. Can you just give me something for it?"
Overly Patient	Verbose, excessive detail, tangential	"Well, good morning to you too, Doctor. It's actually quite interesting how this all started..."
📊 Model Performance
Metric	Value	Notes
Model Size	~300-500MB	After quantization
Training Time	10-30 minutes	On consumer GPU
Inference Speed	<100ms	Per response
Memory Usage	<2GB	During inference
🔧 Technical Details
Model Architecture
Base Model: Microsoft DialoGPT-small or GPT-2

Fine-tuning Method: LoRA (Low-Rank Adaptation)

Quantization: 4-bit precision

Parameters: 2-8 billion parameters (optimized)

Training Specifications
python
# Key training parameters
learning_rate = 1e-4
batch_size = 2
epochs = 3
lora_rank = 4
max_length = 128
Integration Options
Ollama Deployment:

bash
# Convert to GGUF format
python convert_to_gguf.py --model_path ./patient_persona_model

# Create Ollama modelfile
ollama create patient-persona -f Modelfile
FastAPI Web Service:

bash
python api_server.py --port 8000
VR Integration:

python
# Sample integration code
from patient_generator import PatientSimulator

simulator = PatientSimulator()
response = simulator.generate_response(
    persona="anxious",
    doctor_input="What brings you in today?"
)
📁 Project Structure
text
patient-persona-generator/
├── patient_persona_fine_tuning.py  # Main training script
├── requirements.txt                 # Dependencies
├── synthetic_data/                 # Training data
│   ├── conversations.json         # Sample dialogues
│   └── personas.json              # Persona definitions
├── models/                         # Trained models
├── utils/                         
│   ├── data_loader.py             # Data processing
│   └── model_utils.py             # Model utilities
├── examples/                       # Usage examples
├── tests/                         # Unit tests
└── README.md                      # This file
🧪 Testing
Run the test suite to verify functionality:

bash
python -m pytest tests/ -v

# Test specific persona generation
python tests/test_personas.py
🌟 Example Outputs
Input: "What symptoms have you been experiencing?"

Calm Response:
"I've had a persistent dry cough for about two weeks, along with some mild chest discomfort."

Anxious Response:
"Oh, it's been terrible! I have this awful cough that won't go away and I'm really worried it might be something serious like pneumonia!"

Rude Response:
"Are you even listening? I already told the nurse everything. Do I need to repeat myself to everyone here?"

Overly Patient Response:
"Well, it started about two weeks ago, right after my granddaughter's birthday party. She turned seven, you know, and we had this lovely cake with blue frosting - her favorite color. Anyway, the next morning I noticed this tickle in my throat..."

🔮 Future Enhancements
Multi-language support (Hindi, Tamil, etc.)

Emotional tone modulation

Real-time voice synthesis

Expanded medical scenarios

Integration with popular VR platforms

Admin dashboard for persona management

🤝 Contributing
We welcome contributions! Please see our Contributing Guidelines for details.

Fork the repository

Create a feature branch (git checkout -b feature/amazing-feature)

Commit your changes (git commit -m 'Add amazing feature')

Push to the branch (git push origin feature/amazing-feature)

Open a Pull Request

📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
Government of India for supporting medical education initiatives

Hugging Face for the Transformers library

Microsoft Research for DialoGPT models

The open-source AI community

📞 Support
For questions or support:

Create an issue on GitHub

Email: support@medical-ai.org

Documentation: docs.medical-ai.org

🏥 Ethical Considerations
This system is designed for educational purposes only. Always:

Use synthetic data for training

Maintain patient privacy and confidentiality

Supervise AI-generated content with medical professionals

Clearly disclose AI involvement to students
