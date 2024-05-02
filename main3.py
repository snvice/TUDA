import streamlit as st
import torch
from transformers import DistilBertTokenizer
from transformers import DistilBertForSequenceClassification

# Load the pre-trained model and tokenizer
# model = DistilBertForSequenceClassification.from_pretrained('https://github.com/snvice/TUDA/tree/main/distilbert')
# tokenizer = DistilBertTokenizer.from_pretrained('https://github.com/snvice/TUDA/tree/main/tokenizer')


model = DistilBertForSequenceClassification.from_pretrained(
    'https://github.com/snvice/TUDA/tree/main/distilbert',
    use_auth_token=True
)
tokenizer = DistilBertTokenizer.from_pretrained(
    'https://github.com/snvice/TUDA/tree/main/tokenizer',
    use_auth_token=True
)




# Define the label encoder
label_encoder = {'Depression': 0, 'Suicide': 1, 'Drugs': 2, 'Alcohol': 3}
label_decoder = {idx: label for label, idx in label_encoder.items()}

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Sample statements
sample_statements = [
    "I feel so hopeless and worthless. Nothing brings me joy anymore.",
    "The constant sadness and emptiness inside me is unbearable. I don't know how to cope.",
    "Every day is a struggle to get out of bed. The darkness consumes me, and I can't see a way out.",
    "I can't take this pain anymore. The only way out is to end it all.",
    "The world would be better off without me. I'm a burden to everyone.",
    "I've lost the will to live. Dying seems like the only solution.",
    "I can't function without my daily dose. The cravings are too strong to resist.",
    "My life revolves around finding and using drugs. It's a never-ending cycle.",
    "I've lost control over my addiction. It's ruining my life, but I can't stop.",
    "I can't get through the day without a drink. Alcohol is my coping mechanism.",
    "Drinking has become a way of life for me. I can't imagine living without it.",
    "No matter how hard I try, I keep relapsing into alcoholism. It's destroying me."
]

# Streamlit app
st.set_page_config(page_title="Text Classification with DistilBERT", page_icon=":brain:")

st.title("Text Classification with DistilBERT")
st.write("This app uses a fine-tuned DistilBERT model to classify text into four categories: Depression, Suicide, Drugs, and Alcohol.")

# Sample statement gallery
st.subheader("Sample Statement Gallery")
selected_statement = st.selectbox("Select a sample statement", sample_statements)

if st.button("Classify"):
    # Tokenize the selected statement
    inputs = tokenizer(selected_statement, return_tensors="pt", truncation=True).to(device)

    # Get the model output
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)

    # Get the predicted label and probability
    predicted_label_idx = torch.argmax(probabilities, dim=-1).item()
    predicted_label = label_decoder[predicted_label_idx]
    predicted_probability = probabilities[:, predicted_label_idx].item()

    # Display the result
    st.write(f"**Predicted Label:** {predicted_label}")
    st.write(f"**Probability:** {predicted_probability * 100:.0f}%")

# Footer
st.markdown("---")
st.write("Created by Sam")
