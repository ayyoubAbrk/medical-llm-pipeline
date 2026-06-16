from transformers import pipeline

# 1. On change de tâche ("text-generation") et on choisit un vrai mini-LLM
générateur = pipeline(
    "text-generation", 
    model="microsoft/Phi-3-mini-4k-instruct",
    trust_remote_code=False
)

# 2. On lui pose une question ou on lui donne une consigne (en anglais pour ce modèle)
prompt = "Explain what a heart attack is in three simple sentences."

# 3. On génère la réponse
print("L'IA réfléchit...")
resultat = générateur(prompt, max_new_tokens=50, do_sample=True, temperature=0.7,clean_up_tokenization_spaces=True)

# 4. On affiche le texte généré
print("\n--- Réponse de l'IA ---")
print(resultat[0]['generated_text'])