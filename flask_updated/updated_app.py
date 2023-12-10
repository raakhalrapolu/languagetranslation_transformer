from flask import Flask, request, jsonify
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from flask_cors import CORS  # Import CORS

app = Flask(__name__)
CORS(app) 

# Load models and tokenizers for Czech-English, English-Czech, and Ukrainian-English
# Define paths
csen_model_path = 'models/saved_model_csen.pt'
encs_model_path = 'models/saved_model.pt'
ukren_model_path = 'models/saved_model_ukren_epoch4.pt'  # Update this
enukr_model_path = 'models/saved_model_enukr.pt'  # Update this

def load_model(model_path, model):
    # Load the state dictionary
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))

    # Adjust for DataParallel
    if next(iter(state_dict)).startswith('module.'):
        state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    return model



# Load models using utility function
csen_model = load_model(csen_model_path, T5ForConditionalGeneration.from_pretrained('t5-small'))
encs_model = load_model(encs_model_path, T5ForConditionalGeneration.from_pretrained('t5-small'))
ukren_model = load_model(ukren_model_path, T5ForConditionalGeneration.from_pretrained('t5-small'))
enukr_model = load_model(enukr_model_path, T5ForConditionalGeneration.from_pretrained('t5-small'))

# Load tokenizers
csen_tokenizer, encs_tokenizer, ukren_tokenizer, enukr_tokenizer = [T5Tokenizer.from_pretrained('t5-small') for _ in range(4)]

# Set models to evaluation mode
csen_model.eval()
encs_model.eval()
ukren_model.eval()
enukr_model.eval()

@app.route('/translate', methods=['POST'])
def translate():
    request_data = request.json
    text = request_data.get('text', '')
    language_from = request_data.get('language_from', '')
    language_to = request_data.get('language_to', '')
    print(language_from, language_to)

    # Determine which model and tokenizer to use
    if language_from == 'Czech' and language_to == 'English':
        model = csen_model
        tokenizer = csen_tokenizer
    elif language_from == 'English' and language_to == 'Czech':
        model = encs_model
        tokenizer = encs_tokenizer
    elif language_from == 'German' and language_to == 'English':
        model = csen_model
        tokenizer = csen_tokenizer
    elif language_from == 'English' and language_to == 'German':
        model = encs_model
        tokenizer = encs_tokenizer
    elif language_from == 'Ukrainian' and language_to == 'English':
        model = ukren_model
        tokenizer = ukren_tokenizer
    elif language_from == 'English' and language_to == 'Ukrainian':
        model = enukr_model
        tokenizer = enukr_tokenizer
    else:
        return jsonify({'error': 'Unsupported language pair'})

    prompt = f"translate {language_from} to {language_to}: {text}"
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({'translated_text': translated_text})

if __name__ == '__main__':
    app.run(debug=True)

