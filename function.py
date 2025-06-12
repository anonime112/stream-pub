import language_tool_python
import spacy
import textstat
import re

nlp = spacy.load("fr_core_news_sm")
tool = language_tool_python.LanguageTool('fr')


modalites_regex = [
    r"\bsi\b", r"\blorsque\b", r"\baprès\b", r"\bà condition que\b",
    r"\ben cas de\b", r"\bsous réserve\b", r"\bdu moment que\b",
    r"\baussitôt que\b", r"\bdès que\b"
]

verbes_etat = {
    "être", "sembler", "paraître", "devenir", "rester", "demeurer", "avoir l'air"
,
 "passer",
	"constituer",
	"représenter",
"appartenir"
}

def compter_verbes_action_avances(text: str) -> int:
    doc = nlp(text)
    count = 0

    i = 0
    while i < len(doc):
        token = doc[i]

        # Vérifie si c'est un verbe
        if token.pos_ == "VERB":
            # Cas particulier pour l'expression verbale "avoir l'air"
            if token.lemma_ == "avoir" and i + 1 < len(doc) and doc[i + 1].text.lower() == "l'air":
                i += 2  # On saute "avoir" + "l'air", considéré comme verbe d’état
                continue

            lemme = token.lemma_.lower()

            # Si ce n’est pas un verbe d’état
            if lemme not in verbes_etat:
                # Vérifie la présence d'un sujet ou d'un objet
                has_subject = any(child.dep_ == "nsubj" for child in token.children)
                has_object = any(child.dep_ in ("obj", "dobj") for child in token.children)

                if has_subject or has_object:
                    count += 1

        i += 1

    return count


def detecter_modalites(text):
    """
    Détecte les modalités (conditionnelles, temporelles, restrictives) dans un texte.
    Combine une approche par regex et une analyse syntaxique via spaCy.
    
    Retourne :
    - le nombre de modalités détectées
    - la liste des expressions trouvées
    """
    # Analyse avec spaCy
    doc = nlp(text)
    
    # 1. Par expressions régulières
    texte_lower = text.lower()
    termes_trouves = []
    for pattern in modalites_regex:
        matches = re.findall(pattern, texte_lower)
        termes_trouves.extend(matches)
    
    # 2. Par analyse syntaxique (détection des subordonnants conditionnels)
    for token in doc:
        if token.dep_ in ["mark", "advcl"] and token.text.lower() in {"si", "lorsque", "quand", "à condition", "après"}:
            termes_trouves.append(token.text.lower())

    # Nettoyer les doublons
    termes_uniques = list(set(termes_trouves))
    
    return {
        "nbr": len(termes_uniques),
        "liste": termes_uniques,

    } 


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

termes_flous = {
    "approprié", "adéquat", "suffisant", "rapide", "fiable", "efficace",
    "optimisé", "correct", "important", "requis", "significatif", "amélioré",
    "satisfaisant", "pertinent", "conforme", "en temps voulu", "appropriée",
    "adapté", "utile", "standard", "normal", "prévu", "bon", "mauvais", "meilleur",  "correctement", "efficacement", "de manière appropriée", "optimal",
    "pertinent", "approprié", "bon fonctionnement"
}

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
 Mettre en place un dispositif transversal d’identification précoce et de prévention de la fraude. > DACP
 Mettre en place un dispositif d’alerte risque de fraude intégrant les scénarii de fraude et des contrôles applicatifs associés (exemple : vérification des soldes et paramétrage de la séparation des tâches).  > DACP
 Elaborer un plan de formation/sensibilisation les sujets de fraudes dans la stratégie globale de gestion des risques.  > DACP
 Faire appel à un tiers spécialisé pour l’audit de la configuration des critères l'alerte mis en place par le prestataire.  > DACP
"""





def evaluer_comprehension(texte):
    doc = nlp(texte)
    nb_phrases = len(list(doc.sents))
    nb_mots = len(doc)
    moyenne_longueur = nb_mots / nb_phrases if nb_phrases else 1
    lisibilite = textstat.flesch_reading_ease(texte)
    fautes = len(tool.check(texte))

    score = (
        0.4 * (1 - min(moyenne_longueur / 25, 1)) +  # phrases courtes = plus clair
        0.3 * min(lisibilite / 100, 1) +             # meilleure lisibilité
        0.3 * (1 - min(fautes / 5, 1))               # moins de fautes = plus compréhensible
    )
    return max(0.0, min(score, 1.0))



def calculer_densite_ambiguite_lexicale(text: str) -> float:
    """
    Calcule la densité d’ambiguïté lexicale (nb de termes flous / nb total de tokens).
    Retourne une valeur entre 0 et 1 (ou plus si le texte est très court et très flou).
    """
    doc = nlp(text)
    nb_tokens = len(doc)

    if nb_tokens == 0:
        return 0.0

    nb_flous = sum(1 for token in doc if token.lemma_.lower() in termes_flous)
    densite = nb_flous / nb_tokens

    return densite




def extraire_actions_ou_procedures(texte: str) -> list[str]:
    lignes = texte.splitlines()
    actions = []
    complexite=0
    difficulte=0

    # motif pour les sous-tâches comme (i), (ii)
    motif_sous_taches = re.compile(r"\(\s?[ivxlcdm]+\s?\)\s*", re.IGNORECASE)
    # motif pour les bullets
    motif_puce = re.compile(r"^\s*[-•*•\d.]+\s*")
    motif_etape = re.compile(
    r"^(E?tape\s*\d+\s*:?|\bEape\d+\s*:?)",
    re.IGNORECASE
)

    for ligne in lignes:
        ligne_originale = ligne.strip()
        if not ligne_originale:
            continue

        # Gestion des sous-tâches
        if motif_sous_taches.search(ligne_originale):
            sous_taches = re.split(motif_sous_taches, ligne_originale)
            intro = sous_taches[0].strip(": ")
            for sous in sous_taches[1:]:
                if sous.strip():
                    actions.append(f"{intro} : {sous.strip()}")
            continue

        # Suppression de la puce éventuelle
        ligne = motif_puce.sub("", ligne_originale)
        # Suppression de la puce éventuelle
        ligne = motif_etape.sub("", ligne_originale)


        doc = nlp(ligne)
        phrase = list(doc.sents)[0] if list(doc.sents) else nlp(ligne)

        # Cas 1 : Verbe détecté dans la phrase (devoir, infinitif, impératif, etc.)
        if any(
            token.lemma_ == "devoir"
            or "Inf" in token.morph.get("VerbForm")
            or "Imp" in token.morph.get("VerbForm")
            or token.pos_ == "VERB"
            for token in phrase
        ):
            actions.append(ligne)
            continue

        # Cas 2 : Phrase courte commençant par un nom (action implicite)
        tokens = [token for token in phrase if not token.is_punct]
        if tokens and len(tokens) <= 7 and tokens[0].pos_ == "NOUN":
            actions.append(ligne)
            continue
        
    for t in actions:
        complexite = complexite+evaluer_complexite(t)
        difficulte = difficulte+evaluer_difficulte_tache(t)
 
        

    return {
        "longueur_moyenne": actions,
        "nb_phrase":len(actions) ,
         "complexite":complexite,
          "difficulte":difficulte ,
      
    }



def extraire_taches_regex_spacy(text: str) -> list[str]:
    lignes = text.splitlines()
    taches = []

    motif_tache_directe = re.compile(
        r"^(?:-|\*||•|\d+\.)?\s*(?:Mettre en place|Sensibiliser|Faire appel|Définir|Revue|Accroître|Élaborer|Veiller à|Interfacer|Fixer|Établir|Formaliser|Inclure|Créer|Déployer|Embarquer|Revoir|Élargir|Assurer|Renforcer|Doit|Doivent|Devra|Traiter|Procéder|Déterminer)",
        re.IGNORECASE
    )

    motif_sous_taches = re.compile(r"\(\s?[ivxlcdm]+\s?\)\s", re.IGNORECASE)  # (i) (ii) (iii)...

    for ligne in lignes:
        ligne = ligne.strip()
        if not ligne:
            continue

        # Cas 1 : Ligne avec plusieurs sous-tâches entre (i)...(ii)...(iii)...
        if motif_sous_taches.search(ligne):
            sous_taches = re.split(r"\(\s?[ivxlcdm]+\s?\)\s*", ligne)
            phrase_intro = sous_taches[0].strip(": \n")
            for sous in sous_taches[1:]:
                if sous.strip():
                    taches.append(f"{phrase_intro} : {sous.strip()}")
            continue

        # Cas 2 : ligne directe qui commence par un verbe d'action
        if motif_tache_directe.search(ligne):
            taches.append(ligne)
            continue

        # Cas 3 : analyse grammaticale
        doc = nlp(ligne)
        for phrase in doc.sents:
            texte_phrase = phrase.text.strip()
            doc_phrase = nlp(texte_phrase)
            for token in doc_phrase:
                formes = token.morph.get("VerbForm")
                if token.lemma_ == "devoir" or "Inf" in formes or "Imp" in formes:
                    taches.append(texte_phrase)
                    break

    return {
        "longueur_moyenne": taches,
        "nb_phrase":len(taches) ,
      
    }


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



taches1 = extraire_taches_regex_spacy(texte)

for t in taches1:
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
print(len(taches1))
print(taches1["nb_phrase"])
# AnalyseTache(taches)


