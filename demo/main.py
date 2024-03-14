import numpy as np
import pandas as pd
import pickle
import streamlit as st
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import os 


os.environ["OPENAI_API_KEY"] = ''

st.set_page_config(page_title='Mining and Summarizing Customer Interactions and Feedback', layout = 'wide', page_icon = './logo.png')
st.title(':hugging_face: :blue[Mining and Summarizing Customer Interactions and Feedback] :hugging_face:')
# Custom CSS for Styling
st.markdown("""
    <style>
        /* Reduce top margin of the main container */
        .main .block-container {
            padding-top: 2rem; /* Adjust the top padding as needed */
        }
        
        /* Page background color and font color */
        body {
            background-color: #f0f2f6; /* Light gray background */
            color: black; /* All font in black */
            font-size: 18px; /* Font size for body */
        }

        /* Main theme colors and font */
        :root {
            --primary-color: #4CAF50;  /* Green */
            --font: 'Consolas', monospace;  /* Consolas font */
        }

        /* Apply font styles */
        h1, h2, h3, .stButton>button, .stSelectbox {
            font-family: var(--font);
            font-size: 18px; /* Font size for other elements */
        }

        /* Style the title */
        h1 {
            color: black; /* Black color for the header */
            font-size: 24px; /* Slightly larger font size for title */
        }

        /* Style the text area */
        .stTextArea textarea {
            font-family: var(--font);
            font-size: 12px; /* Reduced font size for text area */
            border: none; /* Remove border */
        }

        /* Style the button */
        .stButton>button {
            background-color: black;  /* Black background */
            color: white;  /* White text */
            border: none;  /* No borders */
            font-family: var(--font); /* Use your custom font */
            font-size: 16px; /* Adjust font size as needed */
        }

        /* Button color change on click (active state) */
        .stButton>button:active {
            background-color: gray; /* Gray background when clicked */
            color: white; /* Keep text color white */
        }

        /* Button hover and focus state */
        .stButton>button:hover,
        .stButton>button:focus {
            color: white; /* Keep text color white when hovering/focused */
            background-color: black; /* Maintain background color */
        }

        /* Remove border from text input */
        .stTextInput input {
            border: none; /* Remove border */
        }

        /* Remove text input focus border */
        .stTextInput input:focus {
            outline:none; /* Remove focus outline */
    }
    </style>
    """, unsafe_allow_html=True)

def generate_reviews(context):
    
    role = f"""You are provided with key words and phrases using which you are supposed to generate reviews for a product.
            Use the following guidelines:
            1. Do not use terms like 'this online apparel store'
            2. The review should not talk about the brand of the store. Instead it should only talk about the product.
            3. Do not use colorful language because these reviews are supposed to be written by a regular user online who does not leave extremely verbose reviews.
            4. Keep it simple and authentic as if it is being written a general person online.
            5. Generate at least 1 sentence and at most 7 sentences.

            Sample reviews are below -
            1. Absolutely wonderful - silky and sexy and comfortable
            2. Love this dress!  its sooo pretty.  i happened to find it in a store, and im glad i did bc i never would have ordered it online bc its petite.  i bought a petite and am 58.  i love the length on me- hits just a little below the knee.  would definitely be a true midi on someone who is truly petite.,
            3. Some major design flaws I had such high hopes for this dress and really wanted it to work for me. i initially ordered the petite small (my usual size) but i found this to be outrageously small. so small in fact that i could not zip it up! i reordered it in petite medium, which was just ok. overall, the top half was comfortable and fit nicely, but the bottom half had a very tight under layer and several somewhat cheap (net) over layers. imo, a major design flaw was the net over layer sewn directly into the zipper - it c,
            4. My favorite buy! I love, love, love this jumpsuit. its fun, flirty, and fabulous! every time i wear it, i get nothing but great compliments!,
            5. Flattering shirt This shirt is very flattering to all due to the adjustable front tie. it is the perfect length to wear with leggings and it is sleeveless so it pairs well with any cardigan. love this shirt!!
            """
    question = f"""Use the context: {context}\n and generate a text as if it were a review on a online apparel store's website.
                        """
    client = OpenAI()

    # Call the OpenAI API
    response = client.chat.completions.create(
          model="gpt-4",  # or the relevant GPT-4 engine when available
          messages=[{'role':'system', 'content' : role},
                     {'role': 'user', 'content': question}],
          max_tokens=200  # adjust as needed
        )
    generated_review = response.choices[0].message.content
    return generated_review
    
def summarize(context):
    
    role = f"""You are product review summarizer who helps users to understand what other users have written in reviews.
            Use the following guidelines:
            1. Do not use terms like 'this online apparel store'
            2. The review should not talk about the brand of the store. Instead it should only talk about the product and what aspects of the producth have been highlighted in the reviews.
            3. Do not use colorful language because these reviews are supposed to be written by a regular user online who does not leave extremely verbose reviews.
            4. Keep it simple and authentic as if it is being written a general person online.
            5. Keep the summary, short, crisp and conscise. 

            Below is an example -
            1. I purchased this toy for my 6-year-old daughter and it's been a hit! She absolutely loves the vibrant colors and how it lights up. The fun factor is through the roof – it keeps her engaged for hours. I found the price to be quite reasonable given how much value we've gotten out of it. It's not just a toy; it's become her go-to for playtime. Definitely a performance powerhouse that's worth every penny!
            2. While the toy has a lot of potentials, I'm not convinced it's worth the hefty price tag. My son was initially excited, but the novelty quickly wore off. It's not as intuitive as I expected, and we've struggled with some of the features, making it less used than I had hoped. For a toy at this price point, I expected a bit more durability and easier usage. It's fun, sure, but I wouldn't say it's been a worthwhile investment.
            3. I've got mixed feelings about this toy. On one hand, it's innovative and has some really cool aspects – the tech behind it is impressive, and when it works, it's incredibly fun. On the other hand, the ease of use isn't quite there. It took a lot of time for my kids to figure out how to use it properly, which was a bit disappointing. It's engaging once you get the hang of it, but getting to that point can be a bit of a chore.
            
            Summary: 
            This toy has received positive feedback from customers regarding its fun factor, appearance, value, performance. 
            However, some customers have found this product very expensive for a toy that wasn't worth it.

            """
    question = f"""Use the reviews: {context}\n and generate a summary that captures the different aspects of the reviews. Make sure to refer the above example and maintain  a similar tone and style.
                        """
    client = OpenAI()

    # Call the OpenAI API
    response = client.chat.completions.create(
          model="gpt-4",  # or the relevant GPT-4 engine when available
          messages=[{'role':'system', 'content' : role},
                     {'role': 'user', 'content': question}],
          max_tokens=200  # adjust as needed
        )
    generated_review = response.choices[0].message.content
    return generated_review


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
def encode_sentences(model_name, sentences):
    """
    Reads sentences and gives a embeddings
    """
    if model_name == 'minilmv2': 
        #all MiniLM L6 v2
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

    elif model_name == 'mpnetbase':
        # Load model from HuggingFace Hub
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
        model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # Define a batch size
    batch_size = 32  # Adjust this based on your GPU's memory

        # Function to process batches
    def process_batch(batch):
        encoded_input = tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        
        with torch.no_grad():
            model_output = model(**encoded_input)

        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        return F.normalize(sentence_embeddings, p=2, dim=1)

    # Process in batches
    all_embeddings = []
    for i in range(0, len(sentences), batch_size):
        if i % 1024 == 0: print(f'Done {i}/ {len(sentences)}')
        batch = sentences[i:i+batch_size]
        embeddings = process_batch(batch)
        all_embeddings.append(embeddings)
    # Concatenate all embeddings
    all_embeddings = torch.cat(all_embeddings, dim=0)
    return all_embeddings

def main():
    
        
    with open('./data/corpus_texts.pkl', 'rb') as f:
        corpus_texts = pickle.load(f)
    
    if 'review_text' not in st.session_state:
        st.session_state.review_text = ""
    if 'context' not in st.session_state:
        st.session_state.context = ""
    if 'reviews_list' not in st.session_state:
        st.session_state.reviews_list = ""
    reviews_list = ''
    #c1, title, c3 = st.columns([10,15,10])
    #with title:
    #    st.title("Review Retriever")
    #st.markdown("""<br>""", unsafe_allow_html=True)
    
    
    keywords, spacer, review_col, spacer2, similarities, summary_disp = st.columns([10,3,15,3,15,15])
    with keywords:
        st.subheader("Enter phrase or keywords...")
        user_input_kw = st.text_area("", height=150, label_visibility="collapsed")
        generate_button = st.button('Generate Review')
    with spacer:
        pass
    with review_col:
        st.subheader('Generated Review')
        if generate_button:
            with st.spinner('Generating review using OpenAI...'):
                st.session_state.review_text = generate_reviews(user_input_kw)

        # Use a text area to display the generated review
        st.text_area(label="", value=st.session_state.review_text, height=150, key = 'generated_review', disabled=False)        
        btn_col1, btn_col2 = st.columns([5, 12])
        # Reset button for the generated review
        
        with btn_col1:
            reset_button = st.button("Clear")
            if reset_button:
                st.session_state.review_text = ""
        with btn_col2: 
            embedding_type = st.selectbox("Select the type of embedding",
                                      ('all-MiniLM-L6-v2', 'all-mpnet-base-v2'), 
                                      label_visibility="collapsed")
            submit_button = st.button('Submit')
    with spacer2:
            pass
    with similarities:
        if submit_button:
            # Results are displayed in the second column
            if st.session_state['review_text']:
                # Convert user input to embedding
                if embedding_type == 'all-MiniLM-L6-v2':
                    with st.spinner('Embeddings being generated...'):
                        user_embedding = encode_sentences(model_name = 'minilmv2', sentences = [st.session_state['review_text']])
                    with st.spinner('Loading saved embeddings...'):
                        corpus_embeddings = np.load('./data/allmini_lm_v6_embeddings.npy')
                elif embedding_type == 'all-mpnet-base-v2':
                    with st.spinner('Embeddings being generated...'):
                        user_embedding = encode_sentences(model_name = 'mpnetbase', sentences = [st.session_state['review_text']])
                    with st.spinner('Loading saved embeddings...'):
                        corpus_embeddings = np.load('./data/mpnet_base_embeddings.npy')
                
                print(user_embedding.shape)
                print(corpus_embeddings.shape)
                # Compute similarities
                similarities = cosine_similarity(user_embedding, corpus_embeddings)[0]
                top_indices = similarities.argsort()[-5:][::-1] 

                # Find top similar texts and their indices
                data = {
                    "Text": [corpus_texts[i] for i in top_indices],  # Full text
                    "Cosine Similarity": [round(similarities[i], 4) for i in top_indices]
                }

                # Display results in a formatted way
                st.subheader("Top 5 Similar Reviews:")
                for i in top_indices:
                    st.markdown(f"Score: {similarities[i]:.4f}")
                    st.markdown(f"Text: {corpus_texts[i]}")
                    st.write("---")  # Adds a separator line
                reviews_list = data['Text']
                for i in range(0, len(reviews_list)): 
                    reviews_list[i] = str(i)+ '. ' + reviews_list[i]
                
    with summary_disp: 
        if submit_button:
            context = ' '.join(reviews_list)
            summarized_reviews = summarize(context)
            # Display results in a formatted way
            st.subheader("Summary:")
            st.markdown(summarized_reviews)
                
if __name__ == '__main__':
    main()
