import os
import re
import pytesseract
import torch
from flask import Flask, request, render_template, jsonify
from PyPDF2 import PdfReader
from PIL import Image
from datasets import load_dataset
from transformers import GPTNeoForCausalLM, GPT2Tokenizer, Trainer, TrainingArguments
from stable_baselines3 import PPO
from stable_baselines3.common.envs import DummyVecEnv
from stable_baselines3.common.envs import CustomEnv
from gym import spaces
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'

# GPT-Neo Trainer
class GPTNeoTrainer:
    def __init__(self, model_name="EleutherAI/gpt-neo-1.3B"):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPTNeoForCausalLM.from_pretrained(model_name)

    def train(self, dataset):
        tokenized_dataset = dataset.map(self.tokenize_function, batched=True)
        training_args = TrainingArguments(
            output_dir="./models",
            evaluation_strategy="steps",
            eval_steps=500,
            per_device_train_batch_size=2,
            num_train_epochs=3,
            save_steps=1000,
            save_total_limit=2,
            logging_dir='./logs',
            logging_steps=100,
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset['train'],
            eval_dataset=tokenized_dataset['validation'],
        )
        trainer.train()
        self.model.save_pretrained("./models")

    def tokenize_function(self, examples):
        inputs = [ex["question"] for ex in examples]
        targets = [ex["answer"] for ex in examples]
        model_inputs = self.tokenizer(inputs, padding="max_length", truncation=True, max_length=512)
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(targets, padding="max_length", truncation=True, max_length=512)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def generate_response(self, input_text):
        inputs = self.tokenizer.encode(input_text, return_tensors="pt")
        outputs = self.model.generate(inputs, max_length=150, do_sample=True, top_p=0.95, temperature=0.75)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

# Custom RL environment
class ExamEnv(CustomEnv):
    def __init__(self):
        super(ExamEnv, self).__init__()
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=255, shape=(28, 28, 1), dtype=np.uint8)
        self.state = None
        self.user_feedback = None
        self.done = False

    def step(self, action):
        if action == 0:
            self.user_feedback = 1  # Positive feedback
        elif action == 1:
            self.user_feedback = 0  # Neutral feedback
        elif action == 2:
            self.user_feedback = -1  # Negative feedback
        self.done = True if self.user_feedback is not None else False
        reward = self.user_feedback
        return self.state, reward, self.done, {}

    def reset(self):
        self.state = self._next_observation()
        self.done = False
        return self.state

    def _next_observation(self):
        return np.zeros((28, 28, 1))

def train_rl_agent():
    env = DummyVecEnv([lambda: ExamEnv()])
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)
    return model

# PDF extraction and OCR
def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page_num, page in enumerate(reader.pages):
            text += page.extract_text() or ""
        return text
    except Exception as e:
        return f"Error extracting text from PDF: {e}"

def extract_images_from_pdf(page):
    # Placeholder for future image extraction logic
    return []

def clean_extracted_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

# Preprocess datasets for GPT-Neo training
def preprocess_datasets(datasets):
    preprocessed_data = []
    for entry in datasets['train']:
        question, answer = entry['question'], entry['answers']['text'][0] if entry['answers']['text'] else ""
        if question and answer:
            preprocessed_data.append({"question": question, "answer": answer})
    return preprocessed_data

# Dynamic Chat Functionality
def generate_chat_response(model, user_input):
    response = model.generate_response(user_input)
    return response

# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    if 'pdf' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    pdf = request.files['pdf']
    if pdf.filename == '':
        return jsonify({"error": "No selected file"}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf.filename)
    pdf.save(file_path)

    pdf_text = extract_text_from_pdf(file_path)
    cleaned_text = clean_extracted_text(pdf_text)

    return jsonify({"extracted_text": cleaned_text}), 200

@app.route('/train_model', methods=['POST'])
def train_model():
    datasets = load_dataset('squad')
    preprocessed_data = preprocess_datasets(datasets)
    dataset = load_dataset('json', data_files={'train': preprocessed_data})

    gpt_trainer = GPTNeoTrainer()
    gpt_trainer.train(dataset)

    return jsonify({"message": "Model training complete"}), 200

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get("user_input")
    if not user_input:
        return jsonify({"error": "No input provided"}), 400

    gpt_trainer = GPTNeoTrainer()  # Assuming the model is already trained and loaded
    response = generate_chat_response(gpt_trainer, user_input)
    return jsonify({"response": response}), 200

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
