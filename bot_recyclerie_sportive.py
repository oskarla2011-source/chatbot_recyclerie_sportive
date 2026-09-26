import os
import time
from google import genai
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import streamlit as st
import torch


@st.cache_resource(show_spinner="Veuillez patienter quelques instants, le modèle charge...")
def charger_modele():
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


modele = charger_modele()

# ─────────────────────────────────────────────
# 1. CONFIG PAGE + CSS (Thème Recyclerie Sportive)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Assistant La Recyclerie Sportive",
    page_icon="♻️",
    layout="centered",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@400;500;600&display=swap');
 
/* ── Global ── */
html, body, [data-testid="stAppViewContainer"] {
    background: #f4f7f4 !important;
    font-family: 'DM Sans', sans-serif;
}
#MainMenu, footer, header { visibility: hidden; }
 
/* ── Header ── */
.bs-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 20px 0 14px 0;
    border-bottom: 3px solid #2E7D32;
    margin-bottom: 20px;
}
.bs-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2rem;
    letter-spacing: 2px;
    color: #1b4332;
    line-height: 1;
    margin: 0;
}
.bs-subtitle {
    font-size: 0.72rem;
    color: #52b788;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin: 2px 0 0 0;
}
 
/* ── Badge ── */
.bs-badge {
    display: inline-block;
    background: #fff;
    border: 1px solid #d8f3dc;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.74rem;
    color: #2d6a4f;
    margin-bottom: 16px;
}
.bs-badge b { color: #2E7D32; }
 
/* ── FAQ pills ── */
.bs-faq-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #74c69d;
    margin-bottom: 8px;
}
 
/* ── Messages ── */
[data-testid="stChatMessage"] {
    border-radius: 12px !important;
    margin-bottom: 6px !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    background: #ffffff !important;
    border: 1px solid #e8f5e9 !important;
    border-left: 3px solid #2E7D32 !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: #e8f5e9 !important;
}
 
/* ── Input (Correction bug double contour + bouton vert) ── */
[data-testid="stChatInput"] {
    border: 2px solid #2E7D32 !important;
    border-radius: 12px !important;
    background-color: #ffffff !important;
    box-shadow: none !important;
}

[data-testid="stChatInput"] textarea {
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
    background: transparent !important;
}

/* Bouton d'envoi (flèche) en vert au lieu de rouge */
[data-testid="stChatInput"] button {
    background-color: #2E7D32 !important;
    border: none !important;
    border-radius: 8px !important;
}

[data-testid="stChatInput"] button:hover {
    background-color: #1b4332 !important;
}

[data-testid="stChatInput"] button svg {
    color: #ffffff !important;
    fill: #ffffff !important;
}
 
/* ── Spinner ── */
.bs-thinking {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 0.84rem;
    color: #52b788;
    padding: 6px 0;
}
.bs-dots { display: flex; gap: 5px; }
.bs-dot {
    width: 7px; height: 7px;
    background: #2E7D32;
    border-radius: 50%;
    animation: bs-bounce 1.2s infinite ease-in-out;
}
.bs-dot:nth-child(2) { animation-delay: 0.2s; }
.bs-dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes bs-bounce {
    0%,80%,100% { transform:scale(0.6); opacity:0.3; }
    40%          { transform:scale(1.1); opacity:1;   }
}
 
/* ── Bouton effacer ── */
.stButton > button {
    background: #fff !important;
    border: 1.5px solid #d8f3dc !important;
    border-radius: 8px !important;
    color: #2d6a4f !important;
    font-size: 0.78rem !important;
    padding: 4px 14px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    border-color: #2E7D32 !important;
    color: #2E7D32 !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# 2. CLÉ API
# ─────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error(
        "Clé API non trouvée ! Définis GOOGLE_API_KEY dans ton environnement."
    )
    st.stop()

client = genai.Client(api_key=GOOGLE_API_KEY)
MODEL_NAME = "gemini-2.5-flash"

# ─────────────────────────────────────────────
# 3. CATALOGUE CSV
# ─────────────────────────────────────────────
try:
    df_produits = pd.read_csv("produits_recyclerie_sportive.csv")
except Exception as e:
    st.error(f"Erreur chargement CSV : {e}")
    df_produits = pd.DataFrame()

# ─────────────────────────────────────────────
# 4. BASE DE CONNAISSANCES RAG
# ─────────────────────────────────────────────
KNOWLEDGE_CHUNKS = [
    {
        "id": "presentation",
        "keywords": [
            "recyclerie",
            "association",
            "qui",
            "présentation",
            "concept",
            "écologique",
            "seconde main",
            "sport eco",
        ],
        "text": (
            "La Recyclerie Sportive est une association spécialisée dans le réemploi des équipements sportifs. "
            "Elle collecte, répare et revalorise le matériel de sport d'occasion pour le rendre accessible à tous à petit prix."
        ),
    },
    {
        "id": "dons",
        "keywords": [
            "don",
            "donner",
            "déposer",
            "matériel",
            "collecte",
            "vieux",
            "équipement",
            "donneries",
            "vêtements",
            "vélo",
        ],
        "text": (
            "DONS & COLLECTE :\n"
            "- Vous pouvez déposer vos équipements sportifs usagés ou inutilisés directement dans nos boutiques/ateliers.\n"
            "- Nous acceptons les vélos, vêtements de sport, matériel d'extérieur, ballons, raquettes, etc., quel que soit l'état.\n"
            "- Les articles sont triés, réparés ou revalorisés en atelier solidaire."
        ),
    },
    {
        "id": "ateliers_reparation",
        "keywords": [
            "réparer",
            "réparation",
            "atelier",
            "vélo",
            "auto-réparation",
            "co-réparation",
            "entretenir",
            "mécanique",
        ],
        "text": (
            "ATELIERS DE CO-RÉPARATION :\n"
            "- Nous proposons des ateliers d'auto-réparation (notamment vélo) accompagnés par nos mécaniciens experts.\n"
            "- Des outils et pièces de rechange d'occasion sont mis à disposition des adhérents pour apprendre à entretenir son matériel."
        ),
    },
    {
        "id": "boutiques_et_horaires",
        "keywords": [
            "magasin",
            "boutique",
            "adresse",
            "horaires",
            "ouvert",
            "venir",
            "sur place",
            "localisation",
            "lieu",
        ],
        "text": (
            "BOUTIQUES & ADRESSES :\n"
            "- La Recyclerie Sportive dispose de plusieurs boutiques et ateliers (ex: Massy, Paris 17, Boulevard Bessières, Massy-Palaiseau, etc.).\n"
            "- Les boutiques sont généralement ouvertes du mardi au samedi (consulter le site pour les horaires spécifiques de chaque point de vente).\n"
            "- Possibilité d'acheter sur place ou de retirer ses achats en Click & Collect."
        ),
    },
    {
        "id": "livraison_retrait",
        "keywords": [
            "livraison",
            "envoi",
            "retrait",
            "click and collect",
            "expédition",
            "colissimo",
            "frais de port",
        ],
        "text": (
            "LIVRAISON & RETRAIT :\n"
            "- Retrait gratuit en boutique / atelier (Click & Collect).\n"
            "- Envoi postal disponible pour les petits articles et vêtements.\n"
            "- Pour les équipements volumineux (vélos, machines), le retrait sur place est privilégié."
        ),
    },
    {
        "id": "contact",
        "keywords": [
            "contact",
            "support",
            "email",
            "téléphone",
            "joindre",
            "question",
            "information",
        ],
        "text": (
            "CONTACT :\n"
            "- Directement sur notre site web officiel dans la rubrique Contact.\n"
            "- Directement auprès des équipes en boutique lors des heures d'ouverture."
        ),
    },
]

SYSTEM_PROMPT = (
    "Tu es l'assistant conseil virtuel de La Recyclerie Sportive, une association dédiée au réemploi du matériel de sport. "
    "Ton rôle est d'orienter les usagers vers des équipements de seconde main reconditionnés, de les informer sur les ateliers de co-réparation et les dons. "
    "RÈGLES ABSOLUES : "
    "1. Mémorise TOUTES les informations fournies par l'utilisateur (sport, taille, budget, besoin) et ne les redemande JAMAIS. "
    "2. Pose UNE SEULE question à la fois si une info clé manque pour faire une recommandation. "
    "3. Recommande 1 à 3 produits de seconde main issus du catalogue avec leur prix et URL dès que possible. "
    "4. Adopte un ton chaleureux, écologiquement engagé, accessible et professionnel. "
    "5. Réponds toujours en français sauf si l'utilisateur écrit dans une autre langue. "
    "6. Inclus systématiquement les liens URL des produits recommandés. "
    "7. Si aucun produit ne correspond, explique-le poliment et suggère de passer en boutique ou d'apporter un don. N'invente AUCUN produit. "
    "8. Pour les questions sur les dons, ateliers vélo ou horaires, réponds directement à partir de la base de connaissances."
)

# Suggestions FAQ cliquables
FAQ_SUGGESTIONS = [
    "🚴 Ateliers réparation vélo",
    "🎁 Comment donner du matériel ?",
    "🏪 Où sont les boutiques ?",
    "📦 Click & Collect & Envois",
]


@st.cache_data
def precalculer_embeddings_chunks():
    textes = [c["text"] for c in KNOWLEDGE_CHUNKS]
    return modele.encode(textes, convert_to_tensor=True)


@st.cache_data
def precalculer_embeddings_produits(_df: pd.DataFrame):
    if _df.empty:
        return None
    colonnes = [
        c for c in ["nom", "sport", "marque", "description"] if c in _df.columns
    ]
    textes = _df.apply(
        lambda r: " ".join(str(r.get(c, "")) for c in colonnes), axis=1
    ).tolist()
    return modele.encode(textes, convert_to_tensor=True)


# Précalcul au démarrage (une seule fois)
emb_chunks = precalculer_embeddings_chunks()
emb_produits = precalculer_embeddings_produits(df_produits)


# ─────────────────────────────────────────────
# 5. FONCTIONS RAG
# ─────────────────────────────────────────────
def selectionner_chunks(question: str, max_chunks: int = 3) -> str:
    q = question.lower()

    # Score mots-clés
    scores_kw = [
        sum(1 for kw in chunk["keywords"] if kw in q)
        for chunk in KNOWLEDGE_CHUNKS
    ]

    # Score sémantique
    emb_question = modele.encode(question, convert_to_tensor=True)
    scores_sem = util.cos_sim(emb_question, emb_chunks)[0].tolist()

    # Score hybride
    max_kw = max(scores_kw) if max(scores_kw) > 0 else 1
    scores_hybrides = [
        (0.6 * scores_sem[i] + 0.4 * (scores_kw[i] / max_kw), KNOWLEDGE_CHUNKS[i])
        for i in range(len(KNOWLEDGE_CHUNKS))
    ]
    scores_hybrides.sort(key=lambda x: x[0], reverse=True)
    return "\n\n".join(c["text"] for _, c in scores_hybrides[:max_chunks])


def rechercher_produits(
    question: str, df: pd.DataFrame, max_results: int = 2
) -> pd.DataFrame:
    if df.empty or emb_produits is None:
        return df

    colonnes = [
        c for c in ["nom", "sport", "marque", "description"] if c in df.columns
    ]
    mots = [m for m in question.lower().split() if len(m) > 2]

    # Score mots-clés
    if mots:
        mask = pd.Series([False] * len(df), index=df.index)
        for mot in mots:
            for col in colonnes:
                mask |= df[col].str.lower().str.contains(mot, na=False)
        scores_kw = mask.astype(float)
    else:
        scores_kw = pd.Series([0.0] * len(df), index=df.index)

    # Score sémantique
    emb_question = modele.encode(question, convert_to_tensor=True)
    scores_sem = util.cos_sim(emb_question, emb_produits)[0].tolist()

    # Score hybride
    max_kw = scores_kw.max() if scores_kw.max() > 0 else 1
    df = df.copy()
    df["_score"] = [
        0.6 * scores_sem[i] + 0.4 * (scores_kw.iloc[i] / max_kw)
        for i in range(len(df))
    ]
    return df.nlargest(max_results, "_score").drop(columns=["_score"])


def formater_produits(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    lignes = []
    for _, row in df.iterrows():
        ligne = f"• {row.get('nom','?')} — {row.get('marque','')} — {row.get('prix','?')} €"
        if "url" in row and pd.notna(row["url"]):
            ligne += f" → {row['url']}"
        if "description" in row and pd.notna(row["description"]):
            ligne += f"\n  {str(row['description'])[:120]}"
        lignes.append(ligne)
    return "\n".join(lignes)


def construire_prompt(question: str, df: pd.DataFrame) -> str:
    contexte_manuel = selectionner_chunks(question)
    produits = rechercher_produits(question, df)
    contexte_produits = formater_produits(produits)
    parties = [f"CONTEXTE RECYCLERIE SPORTIVE :\n{contexte_manuel}"]
    parties.append(
        "OBJECTIF : guider l'usager vers du matériel d'occasion ou les services solidaires de l'association."
    )
    if contexte_produits:
        parties.append(f"PRODUITS CORRESPONDANTS :\n{contexte_produits}")
    parties.append(f"USAGER : {question}")
    return "\n\n".join(parties)


# ─────────────────────────────────────────────
# 6. INTERFACE — HEADER
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# 6. INTERFACE — HEADER
# ─────────────────────────────────────────────
# Affiche directement l'image du logo (ajuste la largeur avec 'width')
st.image("logo.png", width=190)

col1, col2 = st.columns([4, 1])
with col1:
    nb = len(df_produits) if not df_produits.empty else 0
    if nb:
        st.markdown(f'<div class="bs-badge">✅ <b>{nb} équipements d\'occasion</b> disponibles</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="bs-badge">⚠️ Catalogue de seconde main indisponible</div>', unsafe_allow_html=True)
with col2:
    if st.button("🗑️ Effacer", key="clear"):
        st.session_state.messages = []
        st.rerun()

# ─────────────────────────────────────────────
# 7. HISTORIQUE
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.markdown(
        '<div class="bs-faq-label">Questions fréquentes</div>',
        unsafe_allow_html=True,
    )
    cols = st.columns(len(FAQ_SUGGESTIONS))
    for i, faq in enumerate(FAQ_SUGGESTIONS):
        with cols[i]:
            if st.button(faq, key=f"faq_{i}", use_container_width=True):
                st.session_state.pending = faq
                st.rerun()

if "pending" in st.session_state:
    msg = st.session_state.pop("pending")
    st.session_state.messages.append({"role": "user", "content": msg})

for m in st.session_state.messages:
    # Choisit l'avatar selon le rôle
    avatar = "logo.png" if m["role"] == "assistant" else "👤"
    with st.chat_message(m["role"], avatar=avatar):
        st.markdown(m["content"])

# ─────────────────────────────────────────────
# 8. SAISIE UTILISATEUR & APPEL API
# ─────────────────────────────────────────────
if prompt := st.chat_input("Posez votre question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"): # <-- Icône utilisateur ici
        st.markdown(prompt)

msgs = st.session_state.messages
if msgs and msgs[-1]["role"] == "user":
    question = msgs[-1]["content"]
    with st.chat_message("assistant", avatar = "logo.png"):

        placeholder = st.empty()
        placeholder.markdown(
            """
<div class="bs-thinking">
    <div class="bs-dots">
        <div class="bs-dot"></div>
        <div class="bs-dot"></div>
        <div class="bs-dot"></div>
    </div>
    Je cherche pour vous...
</div>
""",
            unsafe_allow_html=True,
        )

        historique_texte = " ".join(
            m["content"] for m in msgs if m["role"] == "user"
        )
        full_prompt = construire_prompt(historique_texte, df_produits)

        historique_api = f"{SYSTEM_PROMPT}\n\n"

        for msg in msgs[:-1]:
            role = "Usager" if msg["role"] == "user" else "Assistant"
            historique_api += f"{role} : {msg['content']}\n\n"

        contexte_manuel = selectionner_chunks(question)
        produits = rechercher_produits(question, df_produits)
        contexte_produits = formater_produits(produits)

        historique_api += f"INFOS RECYCLERIE :\n{contexte_manuel}\n\n"
        if contexte_produits:
            historique_api += (
                f"PRODUITS DISPONIBLES :\n{contexte_produits}\n\n"
            )

        historique_api += f"Usager : {question}\nAssistant :"

        answer = None
        for tentative in range(3):
            try:
                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=historique_api,
                )
                answer = response.text
                break
            except Exception as e:
                err_str = str(e).lower()
                if ("503" in err_str or "429" in err_str) and tentative < 2:
                    time.sleep(2**tentative)
                    continue
                answer = "⚠️ Le service est actuellement très sollicité. N'hésite pas à repasser nous voir ou visiter le site web de La Recyclerie Sportive."
                break

        placeholder.empty()
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()