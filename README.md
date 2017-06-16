# sentiment-analysis-nltk
Predict sentiment score with automatic translate technique
Technologies:

1. Python flask <br />
/sentiment <br />
  -Example request <br />
   {"text": "Please, tell me my sentiment"} <br />
  
   -Example response <br />
   { <br />
       "text": "Please, tell me my sentiment", <br />
       "sentiment": "{'compound': 0.6, 'neu': 0.1, 'pos': 0.56,' neg': 0 <br />
   } <br />

/translate <br />
    -Example request <br />
    {"text": "Por favor, traducime al ingles, portugues"} <br />
    
    -Example response <br />
    { <br />
        "text": "Por favor, traducime a 2 idiomas", <br />
        "en": "Please, translate me in two language", <br />
        "pt": "bla bla bla bla" <br />
    }
    


