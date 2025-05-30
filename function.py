import language_tool_python
import spacy
import textstat

nlp = spacy.load("fr_core_news_sm")
tool = language_tool_python.LanguageTool('fr')

mots_abstraits = [
    "gérer", "organiser", "coordonner", "définir", "établir", "optimiser",
    "concevoir", "planifier", "analyser", "mettre en œuvre", "implémenter",
    "structurer", "configurer", "assurer", "piloter", "conduire", "déployer",
    "améliorer", "évaluer", "suivre"
]
verbes_techniques = [
    "compiler", "scripter", "virtualiser", "configurer", "déboguer", "crypter",
    "cloner", "monitorer", "tokeniser", "parser", "indexer", "authentifier"
]
# Liste de verbes abstraits (à enrichir)
verbes_abstraits = set([
    "gérer", "assurer", "optimiser", "superviser", "soutenir", "améliorer",
    "maintenir", "contribuer", "favoriser"
])

# Liste de mots flous (à enrichir aussi)
mots_flous = set([
    "correctement", "efficacement", "de manière appropriée", "optimal",
    "pertinent", "approprié", "bon fonctionnement"
])

verbes_facilitants = [
    "mettre en place", "vérifier", "sécuriser", "formaliser", "documenter",
    "renforcer", "centraliser", "supprimer", "auditer", "remédier", "corriger",
    "justifier", "s'assurer", "établir", "limiter", "instaurer", "réaliser",
    "clarifier", "régulariser", "identifier", "rédiger", "modifier",
    "procéder à", "activer", "concevoir", "diffuser", "revoir", "déployer",
    "mettre à jour", "implémenter", "élaborer", "contrôler", "étayer"
]

ressources_keywords = [
    "ordinateur", "logiciel", "application", "formulaire", "fichier",
    "document", "base de données", "matériel", "réseau", "support",
    "connexion", "email", "identifiant", "dossier", "clé USB", "serveur"
]

subordonnants = [
    "si", "lorsque", "quand", "afin que", "bien que", "parce que", "puisque",
    "avant que", "après que", "pendant que", "comme", "alors que", "tandis que"
]

texte = """
Nous recommandons aux :
1.	Risk Manager de s’assurer que les provisions pour risques et charges intégrées dans l’EP22, sont liées à des cas irréversibles de pertes opérationnelles;
2.	Mettre en place un contrôle conjoint réalisé par le Risk Manager (DRC) et la Direction Financière et comptable (DFC) préalablement à la transmission de l’état prudentiel EP22.
"""


def evaluer_faisabilite(tache):
    a = evaluer_complexite(tache)
    b = evaluer_difficulte_tache(tache)
    a = evaluer_complexite(tache)


def normaliser(val, max_val):
    return max(0, min(100, 100 - (val / max_val) * 100))


def score_comprehension(texte: str) -> dict:
    doc = nlp(texte)
    phrases = list(doc.sents)
    nb_phrases = len(phrases)
    nb_tokens = len([token for token in doc if token.is_alpha])

    longueur_moyenne = nb_tokens / nb_phrases if nb_phrases else 0

    nb_subordonnees = 0
    nb_passives = 0
    for token in doc:
        if "sub" in token.dep_:
            nb_subordonnees += 1
        if token.dep_ == "aux:pass":
            nb_passives += 1

    complexite = nb_subordonnees + nb_passives
    mots_difficiles = textstat.difficult_words(texte)
    score_lisibilite = max(0, min(100, textstat.flesch_reading_ease(texte)))

    # Normalisation (à partir de seuils arbitraires raisonnables)
    score_longueur = normaliser(longueur_moyenne,
                                30)  # seuil max : 30 mots/phrase
    score_complexite = normaliser(complexite,
                                  10)  # seuil max : 10 sub/passives
    score_vocabulaire = normaliser(mots_difficiles,
                                   15)  # seuil max : 15 mots difficiles

    # Moyenne pondérée
    score_global = round((0.25 * score_longueur + 0.25 * score_complexite +
                          0.25 * score_vocabulaire + 0.25 * score_lisibilite),
                         2)

    return {
        "longueur_moyenne": round(longueur_moyenne, 2),
        "nb_subordonnees": nb_subordonnees,
        "nb_passives": nb_passives,
        "mots_difficiles": mots_difficiles,
        "score_lisibilite": score_lisibilite,
        "score_global_comprehension": score_global
    }


def evaluer_comprehension(texte):
    doc = nlp(texte)
    nb_phrases = len(list(doc.sents))
    nb_mots = len(doc)
    moyenne_longueur = nb_mots / nb_phrases if nb_phrases else 1
    lisibilite = textstat.flesch_reading_ease(texte)
    fautes = len(tool.check(texte))

    score = (
        0.4 *
        (1 - min(moyenne_longueur / 25, 1)) +  # phrases courtes = plus clair
        0.3 * min(lisibilite / 100, 1) +  # meilleure lisibilité
        0.3 * (1 - min(fautes / 5, 1))  # moins de fautes = plus compréhensible
    )
    return max(0.0, min(score, 1.0))


def evaluer_complexite(tache):
    doc = nlp(tache)
    nb_mots = len(doc)
    nb_verbes = sum(1 for token in doc if token.pos_ == "VERB")
    nb_subordonnees = sum(1 for token in doc
                          if token.dep_ in ["mark", "advcl", "relcl"])

    nb_abstraits = sum(1 for token in doc if token.lemma_ in mots_abstraits)

    lisibilite = textstat.flesch_reading_ease(tache)
    lisibilite_score = max(0, min(1, (lisibilite - 30) / 70))  # 0-1 inversé

    # Score final normalisé (0 = simple, 1 = complexe)
    score = (
        0.2 * (nb_mots / 20) +  # phrase longue
        0.2 * (nb_verbes / 3) + 0.2 * (nb_subordonnees / 2) + 0.2 *
        (nb_abstraits / 2) + 0.2 *
        (1 - lisibilite_score)  # plus difficile à lire = + complexe
    )
    return score


def AnalyseTache(taches):
    for tache in taches:
        doc = nlp(tache)

        # --- Clarté : test de lisibilité + fautes de grammaire ---
        lisibilite = textstat.flesch_reading_ease(tache)

        fautes = tool.check(tache)
        nb_fautes = len(fautes)
        clarte_score = max(0, min(1, (lisibilite - 30) /
                                  70))  # normalisé entre 0 et 1

        # --- Spécificité : on vérifie le nombre de noms et adjectifs concrets ---
        noms_specifiques = [
            token for token in doc if token.pos_ in ["NOUN", "PROPN", "ADJ"]
        ]
        specificite_score = min(1.0, len(noms_specifiques) / 4)

        # --- Ressources citées : objets, personnes, lieux, outils... ---
        ressources = [
            ent.text for ent in doc.ents
            if ent.label_ in ["MISC", "ORG", "PERSON", "LOC", "PRODUCT"]
        ]
        ressources_score = 1.0 if ressources or any(t.pos_ == "NOUN"
                                                    for t in doc) else 0.0

        print(f"🧾 Tâche : {tache}")
        print(
            f"  - Score de clarté       : {clarte_score:.2f} (Fautes: {nb_fautes})"
        )
        print(
            f"  - Score de spécificité  : {specificite_score:.2f} ({len(noms_specifiques)} éléments)"
        )
        print(
            f"  - Ressources détectées  : {ressources if ressources else 'aucune'} → Score: {ressources_score:.2f}"
        )
        print("")
        acteurs = set()
        actions = set()
        ressources = set()

        for chunk in doc.noun_chunks:
            print("Groupe nominal :", chunk.text)

            texte = chunk.text.strip()

            # Acteurs probables (par heuristique : contient "Manager", "Direction", etc.)
            if any(mot in texte.lower() for mot in
                   ["manager", "direction", "responsable", "service"]):
                acteurs.add(texte)
        # Ressources probables
            elif any(mot in texte.lower() for mot in [
                    "contrôle", "état", "transmission", "provision", "charge",
                    "perte"
            ]):
                ressources.add(texte)

        for token in doc:
            if token.dep_ in ("nsubj", "nmod", "pobj",
                              "appos") and token.pos_ in ("PROPN", "NOUN"):
                acteurs.add(token.text)
            if token.pos_ == "VERB":
                actions.add(token.lemma_)  # le verbe à l'infinitif
            if token.dep_ in ("obj", "obl",
                              "nmod") and token.pos_ in ("NOUN", "PROPN"):
                ressources.add(token.text)

        print("🔸 Phrase analysée :", tache)
        print("👤 Personnes concernées :", ", ".join(acteurs)
              or "Aucune trouvée")
        print("✅ Verbes d’action :", ", ".join(actions) or "Aucun trouvé")
        print("📦 Ressources associées :", ", ".join(ressources)
              or "Aucune trouvée")


def extraire_taches(texte: str) -> list[str]:
    """
    Extrait les tâches d'un texte en détectant les phrases qui contiennent
    des verbes à l'infinitif ou à l'impératif.
    """
    doc = nlp(texte)

    taches = []

    for phrase in doc.sents:
        phrase_doc = nlp(phrase.text)
        for token in phrase_doc:
            if token.pos_ == "VERB" and token.morph.get("VerbForm") in [[
                    "Inf"
            ], ["Imp"]]:
                taches.append(phrase.text.strip())
                break  # une seule détection de verbe suffit

    return taches


def evaluer_difficulte_tache(tache: str) -> dict:
    """
    Évalue la difficulté d'une tâche en retournant un score (0 à 100)
    et les facteurs de difficulté détectés.
    """
    doc = nlp(tache)
    score = 0
    facteurs = []
    texte_min = tache.lower()

    # 1. Mots abstraits et verbes techniques = difficulté ↑
    for token in doc:
        lemma = token.lemma_.lower()
        if lemma in mots_abstraits:
            score += 15
            facteurs.append(f"mot abstrait : {token.text}")
        elif lemma in verbes_techniques:
            score += 20

    # 2. Verbes facilitants = difficulté ↓
    for verbe in verbes_facilitants:
        if verbe in texte_min:
            score -= 10
    # 3. Ressources mentionnées ?
    if not any(r in texte_min for r in ressources_keywords):
        score += 10
    # 4. Subordonnants = complexité syntaxique ↑
    if any(s in texte_min for s in subordonnants):
        score += 10

    # 5. Longueur de la phrase
    if len(doc) > 20:
        score += 10

    # 6. Nombre de mots difficiles (selon textstat)
    nb_difficiles = textstat.difficult_words(tache)
    score += nb_difficiles * 2  # pondération ajustable
    if nb_difficiles > 0:
        facteurs.append(f"{nb_difficiles} mots difficiles")

    difficiles = textstat.difficult_words(tache)
    score = score + difficiles

    return score


taches = extraire_taches(texte)

for t in taches:
    s = evaluer_complexite(t)
    niveau = "Faible" if s < 0.3 else "Moyenne" if s < 0.7 else "Élevée"
    print(f"🧾 Tâche : {t}")
    print(f"  - Score de complexité : {s:.2f} → Niveau : {niveau}\n")
    resultat = evaluer_difficulte_tache(t)
    print(f"  -score_difficulte : {resultat:.2f} → Niveau : {niveau}\n")

    # for m in resultat:
    #   print(f"{m}: {resultat[m]}")

    print(f"------------------------------------------------------------")

score = evaluer_comprehension(texte)
niveau = "Faible" if score < 0.3 else "Moyenne" if score < 0.7 else "Bonne"
print(f"------------------------------------------------------------")
print(f"📝 Compréhension du texte : Score = {score:.2f} → Niveau : {niveau}")
comp = score_comprehension(texte)
for m in comp:
    print(f"{m}: {comp[m]}")
print(f"------------------------------------------------------------")
print(f"------analyse de la tache------")
# AnalyseTache(taches)

