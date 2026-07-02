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
            "Tu es un assistant expert en diagnostic médical. Analyse le document clinique fourni.\n"
            "Génère une réponse structurée en trois parties distinctes :\n"
            "1) SYNTHÈSE : Les symptômes cardinaux identifiés.\n"
            "2) HYPOTHÈSES DIAGNOSTIQUES : Liste les 2 ou 3 pathologies les plus probables au vu des données floues. "
            "Pour chacune, donne un court argument médical (ne donne pas de pourcentage strict).\n"
            "3) RECOMMANDATIONS : Quels examens complémentaires (biologie, imagerie) suggères-tu pour affiner le diagnostic ?"
        )
    },
    {
        "role": "user", 
        "content": f"Voici le document clinique : {text}"
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

answer = result[0]['generated_text'].split("<|assistant|>")[-1]  # Récupération du texte généré par le modèle
print("\n--- Résumé du compte rendu ---")  # Affichage du titre pour le résumé
print(answer.strip())  # Affichage du texte généré par le modèle