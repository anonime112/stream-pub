from numpy import select
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from datetime import datetime
from itertools import islice
from plotly.subplots import make_subplots
import os
import unicodedata
import re
from function import score_comprehension,extraire_taches_regex_spacy,compter_verbes_action_avances,detecter_modalites,calculer_densite_ambiguite_lexicale,extraire_actions_ou_procedures
import io
##################################################################
### Configure App
##################################################################
HISTORIQUE_DIR = "historique"
STATUT_enum = ['', 'En cours ', 'Cloturé', 'Non entamé']
stat_control = ['Non entamé']
stat_contro2 = ['Cloturé']
priorite_enum = ['Importante', 'moyenne', 'Haute']
criticite_enum = ['Elevée', 'Modérée', 'Critique', 'Faible']
colonne = [
    'criticité reco', 'statut de mise en œuvre', 'écheance initiale',
    'écheance revisée', 'priorité pa', '% avancement', 'Date de clôture',
    'ID Recommandation', 'Recommandation', 'Date de clôture', 'Plan dactions ',
    'ID Plan dactions', 'RMO', 'Nb. Reports', 'Nb. Suivis réalisés','Date démission rapport'
]

mapping_criticite= {
    "Modérée": 0,
    "Critique": 1,
    "Elevée": 2,
    "Faible": 3
}
mapping_statut= {
    "En cours": 3,
    "Non entamé": 1,
    "Clôturé": 2,
    
}

def difference_mois(date1, date2):
    """Calcule le nombre de mois entre deux colonnes de dates"""
    return (date2.dt.year - date1.dt.year) * 12 + (date2.dt.month - date1.dt.month)
def difference_jours(date1, date2):
    # Convertir en datetime si ce sont des chaînes
    if isinstance(date1, str):
        date1 = pd.to_datetime(date1, dayfirst=True, errors='coerce')
    if isinstance(date2, str):
        date2 = pd.to_datetime(date2, dayfirst=True, errors='coerce')

    if pd.isnull(date1) or pd.isnull(date2):
        return None  # ou tu peux lever une erreur selon ton besoin

    delta = date2 - date1
    return delta.days


def nombre_trimestres(date_debut, date_fin):
    """
    Calcule le nombre de trimestres entre deux dates.

    Args:
        date_debut (str ou datetime) : Date de début (format 'YYYY-MM-DD' ou datetime)
        date_fin (str ou datetime)   : Date de fin (format 'YYYY-MM-DD' ou datetime)

    Returns:
        int : Nombre de trimestres entre les deux dates
    """
    # Conversion si la date est en string
    if isinstance(date_debut, str):
        date_debut = datetime.strptime(date_debut, "%Y-%m-%d")
    if isinstance(date_fin, str):
        date_fin = datetime.strptime(date_fin, "%Y-%m-%d")

    # Calcul des indices de trimestre (de 0 à 3)
    trimestre_debut = (date_debut.month - 1) // 3
    trimestre_fin = (date_fin.month - 1) // 3

    # Nombre total de trimestres
    nb_trimestres = (date_fin.year - date_debut.year) * 4 + (trimestre_fin - trimestre_debut)

    return nb_trimestres
def comparer_a_aujourdhui(date_str, format_date=None):
    try:
        # Convertir en objet Timestamp de pandas
        if format_date:
            date_donnee = pd.to_datetime(date_str,
                                         format=format_date,
                                         errors='coerce')
        else:
            date_donnee = pd.to_datetime(date_str, errors='coerce')

        if pd.isna(date_donnee):
            return False

        # Date actuelle
        aujourd_hui = pd.Timestamp(datetime.now().date())

        # Comparaison
        if date_donnee.date() < aujourd_hui.date():
            return False
        elif date_donnee.date() > aujourd_hui.date():
            return True
        else:
            return True
    except Exception as e:
        return f"❌ Erreur : {e}"


def convertir_dates(df, col):
    mois_fr_to_en = {
        "janv": "Jan",
        "févr": "Feb",
        "mars": "Mar",
        "avr": "Apr",
        "mai": "May",
        "juin": "Jun",
        "juil": "Jul",
        "août": "Aug",
        "sept": "Sep",
        "oct": "Oct",
        "nov": "Nov",
        "déc": "Dec"
    }

    def remplacer_mois_fr(text):
        if pd.isna(text):
            return text
        for fr, en in mois_fr_to_en.items():
            if fr in text:
                return text.replace(fr, en)
        return text

    # Étape 1 : Nettoyage des mois français
    # df[col + "_cleaned"] = df[col].astype(str).apply(remplacer_mois_fr)

    # Étape 2 : Conversion automatique (avec gestion jour/mois inversés)
    df[col] = pd.to_datetime(df[col + "_cleaned"],
                             dayfirst=True,
                             errors="coerce")

    return df

    # # Remplacement des mois
    # df[col + "_standard"] = df[col].astype(str).apply(remplacer_mois_francais)

    # # Conversion
    # df[col + "_datetime"] = pd.to_datetime(df[col + "_standard"], errors="coerce")
    # return df


def normaliser_texte(texte):
    texte = str(texte)
    if pd.isna(texte):
        return ""
    texte = unicodedata.normalize('NFD', str(texte))
    texte = ''.join(c for c in texte if unicodedata.category(c) != 'Mn')
    texte = re.sub(r"[’'`]", "", texte)  # apostrophes → _
    texte = re.sub(r"[-\s]+", "", texte)  # tirets et espaces → _
    texte = re.sub(r"[^\w_]", "",
                   texte)  # supprimer tout sauf lettres, chiffres, underscore
    return texte.lower()


def remplacer_variantes(df, colonne, enumeration):
    # Créer un mapping normalisé -> forme officielle
    mapping = {normaliser_texte(item): item for item in enumeration}

    def nettoyer_valeur(val):
        val_norm = normaliser_texte(val)
        return mapping.get(
            val_norm,
            val)  # Remplace si correspondance, sinon garde la valeur d'origine

    df[colonne] = df[colonne].apply(nettoyer_valeur)
    return df


def modifier_colonne_si_enum(df, col_ref, col_cible, enum, nouvelle_valeur):
    valeurs_normalisees = {normaliser_texte(e) for e in enum}

    def maj_valeur(row):
        if normaliser_texte(row[col_ref]) in valeurs_normalisees:
            return nouvelle_valeur
        else:
            return row[col_cible]

    df[col_cible] = df.apply(maj_valeur, axis=1)
    return df


if not os.path.exists(HISTORIQUE_DIR):
    os.makedirs(HISTORIQUE_DIR)
st.set_page_config(page_title="Stocks Dashboard", page_icon="💹", layout="wide")
st.html("styles.html")
pio.templates.default = "plotly_white"


# from itertools.batched, used to produce rows of columns
def batched(iterable, n_cols):
    # batched('ABCDEFG', 3) → ABC DEF G
    if n_cols < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n_cols)):
        yield batch


##################################################################
### Data
##################################################################


@st.cache_data
def download_data():
    dfs = pd.read_excel("./Stock Dashboard.xlsx", sheet_name=None)

    history_dfs = {}
    ticker_df = dfs["ticker"].copy(deep=True)

    for ticker in list(ticker_df["Ticker"]):
        d = dfs[ticker]
        history_dfs[ticker] = d

    return ticker_df, history_dfs


@st.cache_data
def transform_data(ticker_df, history_dfs):
    ticker_df["Last Trade time"] = pd.to_datetime(ticker_df["Last Trade time"],
                                                  dayfirst=True)

    for c in [
            "Last Price",
            "Previous Day Price",
            "Change",
            "Change Pct",
            "Volume",
            "Volume Avg",
            "Shares",
            "Day High",
            "Day Low",
            "Market Cap",
            "P/E Ratio",
            "EPS",
    ]:
        ticker_df[c] = pd.to_numeric(ticker_df[c], "coerce")

    for ticker in list(ticker_df["Ticker"]):
        history_dfs[ticker]["Date"] = pd.to_datetime(
            history_dfs[ticker]["Date"], dayfirst=True)

        for c in ["Open", "High", "Low", "Close", "Volume"]:
            history_dfs[ticker][c] = pd.to_numeric(history_dfs[ticker][c])

    ticker_to_open = [
        list(history_dfs[t]["Open"]) for t in list(ticker_df["Ticker"])
    ]
    ticker_df["Open"] = ticker_to_open

    return ticker_df, history_dfs


##################################################################
### App Widgets
##################################################################


def plot_sparkline(data):
    fig_spark = go.Figure(data=go.Scatter(
        y=data,
        mode="lines",
        fill="tozeroy",
        line_color="red",
        fillcolor="pink",
    ), )
    fig_spark.update_traces(hovertemplate="Price: $ %{y:.2f}")
    fig_spark.update_xaxes(visible=False, fixedrange=True)
    fig_spark.update_yaxes(visible=False, fixedrange=True)
    fig_spark.update_layout(
        showlegend=False,
        plot_bgcolor="white",
        height=50,
        margin=dict(t=10, l=0, b=0, r=0, pad=0),
    )
    return fig_spark


def verifier_valeur(val, dtype_attendu):
    if pd.isna(val):
        return False
    if pd.api.types.is_string_dtype(dtype_attendu):
        return isinstance(val, str)
    elif pd.api.types.is_integer_dtype(dtype_attendu):
        return isinstance(
            val,
            int) and not isinstance(val, bool)  # bool est sous-classe de int
    elif pd.api.types.is_float_dtype(dtype_attendu):
        return isinstance(val, (int, float)) and not isinstance(val, bool)
    elif pd.api.types.is_bool_dtype(dtype_attendu):
        return isinstance(val, bool)
    elif pd.api.types.is_datetime64_any_dtype(dtype_attendu):
        return isinstance(val, pd.Timestamp)
    else:
        return True  #


def afficher_types_colonnes_strict(df: pd.DataFrame):
    df_types = df.copy()

    def colorier_cellule(val, col_name):
        dtype_col = df[col_name].dtype
        if pd.isna(val):
            return "background-color: orange;"  # Donnée manquante
        if not verifier_valeur(val, dtype_col):
            return "background-color: red;"  # Mauvais typage
        return ""  # Pas de couleur sinon

    styled_df = df_types.style.apply(
        lambda col: [colorier_cellule(v, col.name) for v in col], axis=0)

    st.write("### 📊 Vérification stricte des types")
    st.dataframe(styled_df, use_container_width=True)


def get_row_count(excel_file):
    df = pd.read_excel(excel_file, sheet_name=0)
    return len(df)


def get_column_count(excel_file):
    df = pd.read_excel(excel_file, sheet_name=0)
    return len(df.columns)


def get_column_info(excel_file):
    df = pd.read_excel(excel_file, sheet_name=0)
    return [(col, str(df[col].dtype)) for col in df.columns]


def analyse_line(excel_file):
    df = pd.read_excel(excel_file, sheet_name=0)
    columns = df.columns


def formulaire_par_colonne(df: pd.DataFrame, ligne_index: int = 0):
    st.write(f"### Formulaire pour la ligne {ligne_index}")

    # S'assurer que l'index est dans le DataFrame
    if ligne_index >= len(df):
        st.error("Index de ligne invalide.")
        return

    ligne = df.iloc[ligne_index]

    for col in df.columns:
        val = ligne[col]
        # st.markdown(f"**{col}** : `{val}`")
        user_input = st.text_input(f"Entrer une valeur pour '{col}'",
                                   value=str(val),
                                   key=col)
        st.write("---")


def ajouter_colonne_calcul(df: pd.DataFrame, nom_colonne="Score Total"):
    # Vérifie que le DataFrame contient au moins 2 colonnes numériques pour l'exemple
    colonnes_numeriques = df.select_dtypes(include='number').columns.tolist()
    # Assumons que les colonnes à additionner sont les 4e et 5e (index 3 et 4 en Python)
    colonne_4 = df.columns[3]
    colonne_5 = df.columns[4]
    # Vérifie que les colonnes existent
    df["Date d'émission rapport"] = pd.to_datetime(df["Date d'émission rapport"], errors="coerce")
    df["Échéance initiale"] = pd.to_datetime(df["Échéance initiale"], errors="coerce")

    # On encode la colonne 'service'
    encodage = pd.get_dummies(df['RMO'], prefix='RMO').astype(int)
    df = pd.concat([df, encodage], axis=1)


    # Calcul de la somme et création d'une nouvelle colonne "Somme"
    df[nom_colonne] = df[colonne_4] + df[colonne_5]
    df["score_comprehension reco"] = df["Recommandation"].apply(
        lambda t: score_comprehension(t)["score_global_comprehension"])
    df["score_comprehension PA"] = df["Plan d'actions "].apply(
        lambda t: score_comprehension(str(t))["score_global_comprehension"] if pd.notnull(t) else 0
    )


    # Calcul vectorisé
    df["nbr_jour_realisation"] = (df["Échéance initiale"] - df["Date d'émission rapport"]).dt.days
    # nbr de jour de cloture
    df["nbr_jour_cloture"] = (df["Date de clôture"] - df["Date d'émission rapport"]).dt.days

    # Appliquer la fonction
    df["nbr_mois"] = difference_mois(df["Date d'émission rapport"], df["Échéance initiale"])

    # Extraire le mois sous forme de nombre (1 à 12)
    df["mois_emission"] = df["Date d'émission rapport"].dt.month
    
    df["nbr_de_tache_reco"] = df["Recommandation"].apply(
        lambda t: extraire_actions_ou_procedures(t)["nb_phrase"])
    df["nbr_de_tache_PA"] = df["Plan d'actions "].apply(
        lambda t: extraire_actions_ou_procedures(str(t))["nb_phrase"])
     
    df["difficulte"] = df["Recommandation"].apply(
        lambda t: extraire_actions_ou_procedures(t)["difficulte"])
    df["complexite"] = df["Plan d'actions "].apply(
        lambda t: extraire_actions_ou_procedures(str(t))["complexite"])
    
    df["nbr_verbe_PA"] = df["Plan d'actions "].apply(
        lambda t: compter_verbes_action_avances((str(t))))
    df["nbr_modalite_PA"] = df["Plan d'actions "].apply(
        lambda t: detecter_modalites(str(t))["nbr"])
    df["nbr_ambiguite_PA"] = df["Plan d'actions "].apply(
        lambda t: calculer_densite_ambiguite_lexicale(str(t)))
    
    
    df["criticite_class"] = df["Criticité reco"].map(mapping_criticite)
    df["statut_class"] = df["Statut de mise en œuvre"].map(mapping_statut)

    # Fonction pour colorier la nouvelle colonne
    def colorier(val):
        return 'background-color: lightblue'  # Colore toute la colonne en bleu clair

    # Appliquer le style sur la colonne "Somme"
    cols = st.columns([6, 1])

    styled_df = df.style.applymap(colorier,
                                  subset=[nom_colonne, "Recommandation","nbr_de_tache_reco",
                                          "nbr_jour_cloture","mois_emission","score_comprehension PA","Nb. Reports"])

    st.success(
        f"Colonne '{[nom_colonne, "Recommandation","nbr_jour_realisation"]}' ajoutée avec la somme de {colonnes_numeriques[:2]}"
    )

    event = st.dataframe(styled_df,
                         key="data",
                         on_select="rerun",
                         selection_mode="single-row")
    event.selection
    selected_row = event.selection
    if selected_row and "rows" in selected_row and len(
            selected_row["rows"]) > 0:
        if selected_row["rows"][0]:
            selected_row_index = selected_row["rows"][0]
            st.write(df.loc[selected_row_index])
            valeur_ville = df.loc[selected_row_index, nom_colonne]
            st.success(
                f"Valeur sélectionnée dans la colonne 'Ville' : {valeur_ville}"
            )

    # edited_df = st.data_editor(styled_df,
    #                            num_rows="dynamic",
    #                            use_container_width=True)

    return df


def display_watchlist_card(ticker, symbol_name, last_price, change_pct, open):
    with st.container(border=True):
        st.html(f'<span class="watchlist_card"></span>')

        tl, tr = st.columns([2, 1])
        bl, br = st.columns([1, 1])

        with tl:
            st.html(f'<span class="watchlist_symbol_name"></span>')
            st.markdown(f"{symbol_name}")

        with tr:
            st.html(f'<span class="watchlist_ticker"></span>')
            st.markdown(f"{ticker}")
            negative_gradient = float(change_pct) < 0
            st.markdown(
                f":{'red' if negative_gradient else 'green'}[{'▼' if negative_gradient else '▲'} {change_pct} %]"
            )

        with bl:
            with st.container():
                st.html(f'<span class="watchlist_price_label"></span>')
                st.markdown(f"Current Value")

            with st.container():
                st.html(f'<span class="watchlist_price_value"></span>')
                st.markdown(f"$ {last_price:.2f}")

        with br:
            fig_spark = plot_sparkline(open)
            st.html(f'<span class="watchlist_br"></span>')
            # st.plotly_chart(
            #     fig_spark, config=dict(displayModeBar=False), use_container_width=True
            # )


def display_watchlist(ticker_df):
    ticker_df = ticker_df[[
        "Ticker", "Symbol Name", "Last Price", "Change Pct", "Open"
    ]]

    n_cols = 4
    for row in batched(ticker_df.itertuples(), n_cols):
        cols = st.columns(n_cols)
        for col, ticker in zip(cols, row):
            if ticker:
                with col:
                    display_watchlist_card(
                        ticker[1],
                        ticker[2],
                        ticker[3],
                        ticker[4],
                        ticker[5],
                    )


def filter_symbol_widget():
    with st.container():
        left_widget, right_widget, _ = st.columns([1, 1, 3])

    selected_ticker = left_widget.selectbox("📰 Currently Showing",
                                            list(history_dfs.keys()))
    selected_period = right_widget.selectbox(
        "⌚ Period", ("Week", "Month", "Trimester", "Year"), 2)

    return selected_ticker, selected_period


def plot_candlestick(history_df):
    f_candle = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.1,
    )

    f_candle.add_trace(
        go.Candlestick(
            x=history_df.index,
            open=history_df["Open"],
            high=history_df["High"],
            low=history_df["Low"],
            close=history_df["Close"],
            name="Dollars",
        ),
        row=1,
        col=1,
    )
    f_candle.add_trace(
        go.Bar(x=history_df.index,
               y=history_df["Volume"],
               name="Volume Traded"),
        row=2,
        col=1,
    )
    f_candle.update_layout(
        title="Stock Price Trends",
        showlegend=True,
        xaxis_rangeslider_visible=False,
        # xaxis=dict(title="date"),
        yaxis1=dict(title="OHLC"),
        yaxis2=dict(title="Volume"),
        hovermode="x",
    )
    f_candle.update_layout(
        title_font_family="Open Sans",
        title_font_color="#174C4F",
        title_font_size=32,
        font_size=16,
        margin=dict(l=80, r=80, t=100, b=80, pad=0),
        height=500,
    )
    f_candle.update_xaxes(title_text="Date", row=2, col=1)
    f_candle.update_traces(selector=dict(name="Dollars"), showlegend=True)
    return f_candle


@st.fragment
def display_symbol_history(ticker_df, history_dfs):
    selected_ticker, selected_period = filter_symbol_widget()

    history_df = history_dfs[selected_ticker]

    history_df["Date"] = pd.to_datetime(history_df["Date"], dayfirst=True)
    history_df = history_df.set_index("Date")
    mapping_period = {"Week": 7, "Month": 31, "Trimester": 90, "Year": 365}
    today = datetime.today().date()
    history_df = history_df[(  today - pd.Timedelta(mapping_period[selected_period], unit="d")):today]

    for c in ["Open", "High", "Low", "Close", "Volume"]:
        history_df[c] = pd.to_numeric(history_df[c])

    left_chart, right_indicator = st.columns([1.5, 1])

    f_candle = plot_candlestick(history_df)

    with left_chart:
        st.html('<span class="column_plotly"></span>')
        st.plotly_chart(f_candle, use_container_width=True)

    with right_indicator:
        st.html('<span class="column_indicator"></span>')
        st.subheader("Period Metrics")
        l, r = st.columns(2)

        with l:
            st.html('<span class="low_indicator"></span>')
            st.metric("Lowest Volume Day Trade",
                      f'{history_df["Volume"].min():,}')
            st.metric("Lowest Close Price", f'{history_df["Close"].min():,} $')
        with r:
            st.html('<span class="high_indicator"></span>')
            st.metric("Highest Volume Day Trade",
                      f'{history_df["Volume"].max():,}')
            st.metric("Highest Close Price",
                      f'{history_df["Close"].max():,} $')

        # with st.container():
        #     st.html('<span class="bottom_indicator"></span>')
        #     st.metric("Average Daily Volume", f'{int(history_df["Volume"].mean()):,}')
        #     st.metric(
        #         "Current Market Cap",
        #         "{:,} $".format(
        #             ticker_df[ticker_df["Ticker"] == selected_ticker][
        #                 "Market Cap"
        #             ].values[0]
        #         ),
        #     )


def display_overview(ticker_df):

    def format_currency(val):
        return "$ {:,.2f}".format(val)

    def format_percentage(val):
        return "{:,.2f} %".format(val)

    def format_change(val):
        return "color: red;" if (val < 0) else "color: green;"

    def apply_odd_row_class(row):
        return [
            "background-color: #f8f8f8" if row.name % 2 != 0 else ""
            for _ in row
        ]

    with st.expander("📊 Stocks Preview"):
        styled_dataframe = (ticker_df.style.format({
            "Last Price":
            format_currency,
            "Change Pct":
            format_percentage,
        }).apply(apply_odd_row_class, axis=1).map(format_change,
                                                  subset=["Change Pct"]))

        st.dataframe(
            styled_dataframe,
            column_order=[column for column in list(ticker_df.columns)],
            column_config={
                "Open":
                st.column_config.AreaChartColumn(
                    "Last 12 Months",
                    width="large",
                    help="Open Price for the last 12 Months",
                ),
            },
            hide_index=True,
            height=250,
            use_container_width=True,
        )


def afficher_tableau_avec_checkbox(df: pd.DataFrame):
    st.write("## 📊 Tableau avec sélection par checkbox")

    # Ajoute une colonne de checkbox à afficher
    selection = []
    selected_index = None

    # Création du tableau avec une checkbox par ligne
    for i in df.index:
        col1, col2 = st.columns([1, 10])
        with col1:
            checked = st.checkbox("", key=f"check_{i}")
        with col2:
            st.write(df.loc[i:i])  # Affiche la ligne comme tableau

        # Ajoute l'index si coché
        if checked:
            selection.append(i)

    # S'assurer qu'une seule ligne est sélectionnée
    if len(selection) > 1:
        st.warning(
            "❗ Vous ne pouvez sélectionner qu'une seule ligne à la fois.")
    elif len(selection) == 1:
        selected_index = selection[0]
        st.success(f"Ligne sélectionnée (index {selected_index}) :")
        st.write(df.loc[[selected_index]])

    return selected_index, df.loc[
        selected_index] if selected_index is not None else None


##################################################################
### Main App
##################################################################

ticker_df, history_dfs = download_data()
ticker_df, history_dfs = transform_data(ticker_df, history_dfs)
all_symbols = list(ticker_df["Ticker"])

st.html('<h1 class="title">Stocks Dashboard</h1>')
st.sidebar.success("Select a demo above.")

display_watchlist(ticker_df)

uploaded_files = st.file_uploader("Choose a CSV file",
                                  accept_multiple_files=False)

if uploaded_files is None:
    st.info("pas de fichier ajouter ")
    st.stop()

# for uploaded_file in uploaded_files:
#     bytes_data = uploaded_file.read()
#     st.write("filename:", uploaded_file.name)
#     dfs = pd.read_excel(bytes_data)
#     st.dataframe(dfs)
# st.divider()
if st.button("enregistrer ", type="primary") and uploaded_files is not None:
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uploaded_files.name}"
    filepath = os.path.join(HISTORIQUE_DIR, filename)
    with open(filepath, "wb") as f:
        f.write(uploaded_files.getbuffer())
    st.success(f"✅ Fichier enregistré sous : {filename}")

if uploaded_files is not None:
    dfs = pd.read_excel(uploaded_files)
    st.dataframe(dfs)
    st.write(f"Nombre de lignes : {get_row_count(uploaded_files)}")
    st.write(f"Nombre de colonnes : {get_column_count(uploaded_files)}")

    st.subheader("Colonnes et types de données :")
    with st.expander("list"):
        for name, dtype in get_column_info(uploaded_files):
            nom_normalise = normaliser_texte(name)

            if nom_normalise in [normaliser_texte(col) for col in colonne]:
                st.markdown(
                    f"<span style='color:#FF8C00'><b>{name}</b> : {dtype}</span>",
                    unsafe_allow_html=True)
            else:
                st.write(f"- **{name}** : {dtype}")
        # for name, dtype in get_column_info(uploaded_files):
        #     st.write(f"- **{name}** : {dtype}")

    # with st.expander("list"):
    #     afficher_types_colonnes(dfs)

    with st.expander("list"):
        afficher_types_colonnes_strict(dfs)

        st.subheader("nettoyage de donnees  :")


        try:
            dfs["Échéance initiale"] = pd.to_datetime(dfs["Échéance initiale"], errors="coerce", dayfirst=True)  # ou dayfirst=False selon le format
            st.success("Colonne 'Date' convertie en datetime.")
            afficher_types_colonnes_strict(dfs)
        except Exception as e:
            st.error(f"Erreur de conversion : {e}")
        message = st.text_area("Message",
                               value="Lorem ipsum.\nStreamlit is cool.")
        if st.button("Prepare download"):
            st.download_button(
                label="Download text",
                data=message,
                file_name="message.txt",
                on_click="ignore",
                type="primary",
                icon=":material/download:",
            )

    # Vérification
    # Normaliser les colonnes du DataFrame et celles attendues
    colonnes_df_normalisees = [normaliser_texte(col) for col in dfs.columns]
    colonnes_requises_normalisees = [normaliser_texte(col) for col in colonne]

    # Vérification des colonnes absentes
    colonnes_absentes = [
        col for col in colonnes_requises_normalisees
        if normaliser_texte(col) not in colonnes_df_normalisees
    ]

    if colonnes_absentes:
        st.error(
            f"Les colonnes suivantes sont manquantes : {', '.join(colonnes_absentes)}"
        )
    else:
        # ✅ Toutes les colonnes sont présentes, on peut exécuter du code
        st.success("Toutes les colonnes nécessaires sont présentes.")
        df = remplacer_variantes(dfs, 'criticité reco', criticite_enum)
        df2 = remplacer_variantes(df, 'priorité pa', priorite_enum)
        df3 = remplacer_variantes(df2, 'priorité pa', STATUT_enum)
        df4 = convertir_dates(df3, "écheance initiale")
        df5 = modifier_colonne_si_enum(df4, "statut de mise en oeuvre",
                                       "% avancement", stat_contro2, "100%")
        df5 = modifier_colonne_si_enum(df5, "statut de mise en oeuvre",
                                       "% avancement", stat_control, "0%")

st.markdown("""
            
            
      
            """)

st.subheader("analyse  de donnees  :")

ddf = ajouter_colonne_calcul(dfs)

# Création du fichier Excel dans un buffer mémoire
output = io.BytesIO()
with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
    ddf.to_excel(writer, index=False, sheet_name='donnees')

# Remettre le curseur du buffer au début
output.seek(0)
# Convertir en CSV avec UTF-8 BOM + ; comme séparateur
# csv_buffer = io.StringIO()
# ddf.to_csv(csv_buffer, sep=';', index=False, encoding='utf-8-sig')
# csv_bytes = csv_buffer.getvalue().encode('utf-8-sig')

# Bouton de téléchargement
st.download_button(
    label="📥 Télécharger le fichier CSV",
    data=output,
    file_name="donnees_calculees.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

with st.container(border=True):
    st.html(f'<span class="watchlist_analyse"></span>')

    tl, tr = st.columns([1, 1])
    bl, br = st.columns([1, 1])
    with tl:
        dfs = pd.read_excel(uploaded_files)
        formulaire_par_colonne(dfs)

display_symbol_history(ticker_df, history_dfs)
# display_overview(ticker_df)
