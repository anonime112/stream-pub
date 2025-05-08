import streamlit as st
import pandas as pd
import os
from urllib.error import URLError

st.set_page_config(page_title="DataFrame Demo", page_icon="📊", layout="wide")

st.markdown("# DataFrame Demo")
st.sidebar.header("DataFrame Demo")
st.write(
    """This demo shows how to use `st.write` to visualize Pandas DataFrames.
(Data courtesy of the [UN Data Explorer](http://data.un.org/Explorer.aspx).)"""
)




st.subheader("Historique des fichiers analyser",divider=True)

HISTORIQUE_DIR = "historique"
if not os.path.exists(HISTORIQUE_DIR):
    st.warning("Aucun fichier trouvé.")
else:
    files = sorted(os.listdir(HISTORIQUE_DIR), reverse=True)
    if files:
        for file in files:
            cl1, cl2=st.columns([1, 1])
            with cl1:
                file_path = os.path.join(HISTORIQUE_DIR, file)
                st.write(f"📄 {file}")
            
            with cl2:
                
                with open(file_path, "rb") as f:
                    st.download_button(label="📥 Télécharger", data=f, file_name=file)
    else:
        st.info("Aucun fichier analysé pour le moment.")
        
        
        
        
st.subheader("Historique des fichiers traiter ")