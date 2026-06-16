import os
from transformers import pipeline
from pypdf import PdfReader

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

prompt = "Résume le compte rendu suivant en un seul paragraphe concis et clair : " + text  # Création du prompt pour le modèle en ajoutant le texte extrait du PDF

print("L'IA réfléchit...")  # Message indiquant que le modèle est en train de générer une réponse
result = generator(
    prompt, 
    max_new_tokens=300,   # Amplement suffisant pour un résumé médical
    max_lenght = None,  # Pas de limite stricte sur la longueur du texte généré
    do_sample=True, 
    temperature=0.1,      # Très basse pour forcer l'IA à rester factuelle et sérieuse
    clean_up_tokenization_spaces=True
)  # Génération de la réponse du modèle avec les paramètres spécifiés

print("\n--- Résumé du compte rendu ---")  # Affichage du titre pour le résumé
print(result[0]['generated_text'])  # Affichage du texte généré par le modèle