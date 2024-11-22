import streamlit as st
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import AutoTokenizer, AutoModelForTokenClassification

st.set_page_config(page_title="NLP Super Application", layout="wide")


TASKS = {
    "Home": "Welcome to the NLP Super Application! Select a task from the left to get started.",
    "Sentiment Analysis": "Analyze the sentiment of your text (positive, negative, or neutral).",
    "Named Entity Recognition": "Extract entities like names, dates, and locations from text.",
    "Summarization": "Summarize long pieces of text into concise summaries.",
    "Text Generation": "Generate creative text based on a prompt.",
    "Machine Translation": "Translate text between languages.",
    "Question Answering": "Get answers to questions from a given context.",
    "Text Similarity": "Find semantic similarity between two texts.",
    "Zero-Shot Classification": "Classify text into unseen categories.",
    "Keyword Extraction": "Extract important keywords or phrases from text.",
    "Emotion Detection": "Identify and classify the emotional tone (e.g., joy, sadness, anger, fear, surprise, love) in a given text.",
}


EXAMPLES = {
    "Sentiment Analysis": "I love the simplicity of this app. It’s amazing!",
    "Named Entity Recognition": "Barack Obama was born on August 4, 1961, in Honolulu, Hawaii.",
    "Summarization": "In the world of technology, artificial intelligence (AI) is a rapidly growing field that has the potential to revolutionize many industries, including healthcare, finance, and education. AI refers to the simulation of human intelligence processes by machines, particularly computer systems. These processes include learning, reasoning, problem-solving, perception, and language understanding. In healthcare, AI can be used to analyze medical data, predict patient outcomes, and assist in diagnosing diseases. In finance, AI algorithms are used for fraud detection, risk management, and personalized financial advice. In education, AI-powered tools are being developed to provide personalized learning experiences for students. Despite its potential, there are challenges and concerns related to the widespread adoption of AI, including data privacy, ethical implications, and the impact on jobs. While AI can bring significant benefits, it is essential to address these concerns to ensure that its integration into society is done in a responsible and ethical manner. As AI continues to evolve, its role in shaping the future of various sectors will only increase, making it one of the most significant technological advancements of the 21st century.",
    "Text Generation": "Once upon a time, in a faraway land, there was a magical forest where",
    "Machine Translation": "Hello, how are you?",
    "Question Answering": {"context": "The Eiffel Tower is located in Paris, France.", "question": "Where is the Eiffel Tower located?"},
    "Text Similarity": ["The quick brown fox jumps over the lazy dog.", "A fast brown fox leaps over a sleeping dog."],
    "Zero-Shot Classification": {"text": "I need to book a flight ticket for my vacation.", "labels": "travel, finance, education"},
    "Keyword Extraction": "Natural Language Processing enables machines to understand human language better.",
    "Emotion Detection": "I feel so excited about this new opportunity!"
}
def count_words(text):
    words = text.split()
    return len(words)

left_column, right_column = st.columns([2, 12])  # Adjust column width as needed



# Left column (for task options)
with left_column:
    st.sidebar.title("✨ NLP Tasks ✨")
    st.sidebar.write("Choose a task to get started:")
    task_icons = {
        "Sentiment Analysis": "😃",
        "Named Entity Recognition": "📍",
        "Summarization": "📝",
        "Text Generation": "💬",
        "Machine Translation": "🌐",
        "Question Answering": "❓",
        "Text Similarity": "🔍",
        "Zero-Shot Classification": "⚖️",
        "Keyword Extraction": "🔑",
        "Emotion Detection": "😞😠🙂",
    }

    selected_task = st.sidebar.radio("Select a Task", list(TASKS.keys()), format_func=lambda task: f"{task_icons.get(task, '')} {task}")



# Right column (only About Me on Home page)
with right_column:
    if selected_task == "Home":
        st.markdown("# Welcome to Your NLP Application! 👋")
        st.markdown("### About Me 🧑‍💻")
        st.write("**Name:** Shaik Hidaythulla 🤖")  
        st.write("**Role:** Data Scientist / NLP Enthusiast 📊 / 🤖")  
        st.write("**Email:** shaikhidaythulla07@gmail.com 📧") 
        st.write("[GitHub](https://github.com/SHAIK-07) 🔗")
        st.write("[LinkedIn](https://www.linkedin.com/in/shaik-hidaythulla/) 🔗")  
        st.markdown("### About Project 📚")
        st.write("This is a comprehensive NLP application that offers a wide range of text processing tasks including sentiment analysis, paraphrasing, keyword extraction, and more. 🔍📝")

    elif selected_task in TASKS:
        task_emoji = task_icons.get(selected_task, "🔧")  # Default to wrench if no emoji is found
        st.markdown(f"### {task_emoji} {selected_task}")
        st.write(TASKS[selected_task])
    

    try:
        # Sentiment Analysis
        if selected_task == "Sentiment Analysis":
            text = st.text_area("Enter text to analyze sentiment:", value=EXAMPLES["Sentiment Analysis"])
            if st.button("Analyze Sentiment"):
                sentiment_analyzer = pipeline("sentiment-analysis")
                result = sentiment_analyzer(text)
                sentiment_label = result[0]['label']
                confidence = result[0]['score']
                sentiment_to_emoji = {
                    "POSITIVE": "😊",  
                    "NEGATIVE": "😞",  
                    "NEUTRAL": "😐"    
                    }
                emoji = sentiment_to_emoji.get(sentiment_label, "🤔") 
       
                st.write(f"**Sentiment:** {sentiment_label} {emoji}")
                st.write(f"**Confidence:** {confidence:.2f}")

                
        
        # Named Entity Recognition
        elif selected_task == "Named Entity Recognition":
        
            text = st.text_area("Enter text to extract named entities:", value=EXAMPLES["Named Entity Recognition"])
    
        
            entity_mapping = {
                "PER": "Person",
                "ORG": "Organization",
                "LOC": "Location",
                "MISC": "Miscellaneous",
                "GPE": "Geopolitical Entity",
                "FAC": "Facility",
                "TIME": "Time",
                "DATE": "Date",
                "PERCENT": "Percentage",
                "MONEY": "Money",
                "QUANTITY": "Quantity",
                "ORDINAL": "Ordinal",
                "CARDINAL": "Cardinal"
                 }

            entity_to_emoji = {
                "Person": "🧑‍🤝‍🧑",   
                "Organization": "🏢",    
                "Location": "🌍",        
                "Miscellaneous": "🔮",   
                "Geopolitical Entity": "🌍",  
                "Facility": "🏥",        
                "Time": "⏰",            
                "Date": "📅",            
                "Percentage": "💯",      
                "Money": "💰",           
                "Quantity": "🔢",        
                "Ordinal": "🔢",         
                "Cardinal": "🔢"         
    }
        
            if st.button("Extract Entities"):
                ner = pipeline("ner", aggregation_strategy="simple")
                result = ner(text)
                st.write("**Extracted Entities:**")
                for entity in result:
                    entity_type_full = entity_mapping.get(entity['entity_group'], entity['entity_group'])
                    entity_emoji = entity_to_emoji.get(entity_type_full, "❓")  
                    st.write(f"{entity['word']} ({entity_type_full} {entity_emoji})")
        
        # Summarization
        elif selected_task == "Summarization":
            text = st.text_area("Enter text to summarize:", value=EXAMPLES["Summarization"])
            word_count = count_words(text)
            st.write(f"**Word Count:** {word_count} words")
            st.write("### Parameters:")
            max_length = st.slider(
                "Maximum Summary Length",
                min_value=20, max_value=200, value=100,
                help="The longest the summary can be. Increase for more detailed summaries."
            )
            min_length = st.slider(
                "Minimum Summary Length",
                min_value=10, max_value=100, value=30,
                help="The shortest the summary can be. Increase for concise summaries."
            )
            if st.button("Summarize"):
                summarizer = pipeline("summarization")
                result = summarizer(text, max_length=max_length, min_length=min_length)
                st.write(f"**Summary:**\n{result[0]['summary_text']}")
                After_word_count = count_words(result[0]['summary_text'])
                st.write(f"**After Summarization Word Count:** {After_word_count} words")
                
        
        # Text Generation
        elif selected_task == "Text Generation":
            prompt = st.text_area("Enter text to generate from:", value=EXAMPLES["Text Generation"])
            st.write("### Parameters:")
            max_length = st.slider(
                "Maximum Length of Generated Text",
                min_value=20, max_value=200, value=50,
                help="Controls how long the generated text can be. Increase for more content."
            )
            temperature = st.slider(
                "Creativity Level (Temperature)",
                min_value=0.1, max_value=1.5, value=1.0,
                help="Higher values make text more creative but less predictable. Lower values make text more focused."
            )
            top_k = st.slider(
                "Top-k Sampling (Number of Choices)",
                min_value=1, max_value=100, value=50,
                help="Limits the number of words considered at each generation step. Lower values lead to more focused output."
            )
            if st.button("Generate Text"):
                generator = pipeline("text-generation", model="gpt2")
                result = generator(prompt, max_length=max_length, temperature=temperature, top_k=top_k)
                st.write(f"**Generated Text:**\n{result[0]['generated_text']}")

        
        # Machine Translation
        elif selected_task == "Machine Translation":
            
            text = st.text_area("Enter text to translate:", value=EXAMPLES["Machine Translation"])
    
            language_map = {
                "French": "fr",
                "German": "de"
                }
    
            
            target_lang = st.selectbox(
                "Select target language (Currently we have French and German only)", list(language_map.keys()),
                help="Choose the language you want to translate your text into."
                )
    
            
            if st.button("Translate Text"):
            
                target_lang_code = language_map[target_lang]
        
        
                pipeline_name = f"translation_en_to_{target_lang_code}"  
                translator = pipeline(pipeline_name)
                result = translator(text)
                st.write(f"**Translated Text:** {result[0]['translation_text']}")
            

        
        # Question Answering
        elif selected_task == "Question Answering":
            context = st.text_area("Enter context:", value=EXAMPLES["Question Answering"]["context"])
            question = st.text_input("Enter question:", value=EXAMPLES["Question Answering"]["question"])
            st.write("### Parameters:")
            top_k = st.slider(
                "Number of Answers to Retrieve",
                min_value=1, max_value=5, value=1,
                help="Retrieve the top-k most relevant answers. Higher values provide more options."
            )
            if st.button("Get Answer"):
                qa = pipeline("question-answering")
                result = qa(question=question, context=context, top_k=top_k)
                if top_k == 1:
                    st.write(f"**Answer:** {result['answer']}")
                else:
                    for i, ans in enumerate(result):
                        st.write(f"**Answer {i+1}:** {ans['answer']}, **Score:** {ans['score']:.2f}")

        
        # Text Similarity
        elif selected_task == "Text Similarity":
            text1 = st.text_area("Enter first text:", value=EXAMPLES["Text Similarity"][0])
            text2 = st.text_area("Enter second text:", value=EXAMPLES["Text Similarity"][1])
            if st.button("Check Similarity"):
                similarity = pipeline("zero-shot-classification")
                result = similarity(text1, candidate_labels=[text2])
                st.write(f"**Similarity Score:** {result['scores'][0]:.2f}")
        
        # Zero-Shot Classification
        elif selected_task == "Zero-Shot Classification":
            text = st.text_area("Enter text for classification:", value=EXAMPLES["Zero-Shot Classification"]["text"])
            labels = st.text_input(
                "Enter Candidate Labels (Comma-Separated):",
                value=EXAMPLES["Zero-Shot Classification"]["labels"],
                help="Provide a list of possible categories for classification, separated by commas."
            )
            multi_label = st.checkbox(
                "Allow Multiple Labels",
                value=False,
                help="Check this if the text can belong to more than one category."
            )
            if st.button("Classify"):
                zero_shot = pipeline("zero-shot-classification")
                result = zero_shot(text, candidate_labels=labels.split(","), multi_label=multi_label)
                for label, score in zip(result["labels"], result["scores"]):
                    st.write(f"**Label:** {label}, **Score:** {score:.2f}")
        
        
        # Keyword Extraction
        elif selected_task == "Keyword Extraction":
            text = st.text_area("Enter text to extract keywords:", value=EXAMPLES["Keyword Extraction"])
            if st.button("Extract Keywords"):
                try:
                    model_name = "ml6team/keyphrase-extraction-distilbert-inspec"
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModelForTokenClassification.from_pretrained(model_name)
                    extractor = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
                    result = extractor(text)
                    keywords = [item["word"] for item in result]
                    st.write("**Extracted Keywords:**")
                    st.write(", ".join(keywords) if keywords else "No keywords found.")
                except Exception as e:
                    st.error(f"An error occurred during keyword extraction: {str(e)}")

        
        # Emotion Detection
        elif selected_task == "Emotion Detection":
            text = st.text_area("Enter text to analyze emotions:", value=EXAMPLES["Emotion Detection"])
            emotion_to_emoji = {
                "anger": "😡",       
                "fear": "😨",        
                "joy": "😊",         
                "love": "😍",        
                "sadness": "😢",      
                "surprise": "😲",     
                "neutral": "😐"}
    
    
            if st.button("Detect Emotion"):
                try:
            
                    emotion_model = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")
                    result = emotion_model(text, top_k=None)

                    st.write("**Detected Emotions with Confidence Scores:**")
                    for item in result:
                        emotion = item['label']
                        emoji = emotion_to_emoji.get(emotion, "❓")  
                        st.write(f"- {emotion.capitalize()} {emoji}: {item['score']:.2f}")
                except Exception as e:
                    st.error(f"An error occurred while detecting emotions: {str(e)}")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
