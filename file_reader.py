from pypdf import PdfReader
import os

reader = PdfReader("cr.pdf") # Initialisation du lecteur PDF avec le fichier "cr.pdf"
text = "" # Initialisation d'une variable pour stocker le texte extrait

for page in reader.pages: # Parcours de chaque page du fichier .pdf
    text_page = page.extract_text() # On extrait le texte de la page
    if text_page: # Si le texte de la page n'est pas vide, on l'ajoute à la variable text
        text += text_page
    

print(text)
