import os
from transformers import pipeline
from pypdf import PdfReader
import warnings
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")

##__________________ INITIALISATION DES VARIABLES __________________##
reader = PdfReader("cr.pdf")  # Initialisation du lecteur PDF avec le fichier "cr.pdf"
text = ""  # Initialisation d'une variable pour stocker le texte extrait

generator = pipeline("text-generation", # initialisation du générateur de texte avec le modèle "microsoft/Phi-3-mini-4k-instruct"
                     model="microsoft/Phi-3-mini-4k-instruct",
                     trust_remote_code=False) 

for page in reader.pages:  # Parcours de chaque page du fichier .pdf
    text_page = page.extract_text()  # On extrait le texte de la page
    if text_page:  # Si le texte de la page n'est pas vide, on l'ajoute à la variable text
        text += text_page

messages = [
    {
        "role": "system", 
        "content": (
            "Tu es un médecin vulgarisateur. Résume le document fourni en 3 phrases courtes "
            "et simples maximum. Sois direct, ne fais pas d'introduction (ne dis pas 'Voici le résumé') "
            "et va droit au but. Arrête-toi immédiatement après le point final de la 3ème phrase."
        )
    },
    {
        "role": "user", 
        "content": f"Voici le document : {text}"
    }
]

prompt_final = generator.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

print("L'IA réfléchit...")  # Message indiquant que le modèle est en train de générer une réponse
result = generator(
    prompt_final, 
    max_new_tokens=300,   # Amplement suffisant pour un résumé médical
    max_length = None,  # Pas de limite stricte sur la longueur du texte généré
    do_sample=True, 
    temperature=0.3,      # Très basse pour forcer l'IA à rester factuelle et sérieuse
    clean_up_tokenization_spaces=True
)  # Génération de la réponse du modèle avec les paramètres spécifiés

print("\n--- Résumé du compte rendu ---")  # Affichage du titre pour le résumé
print(result[0]['generated_text'])  # Affichage du texte généré par le modèle