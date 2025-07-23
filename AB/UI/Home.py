#MLX Week 6.py

import streamlit as st



pages={
    "Projects": [
        st.Page("MoMoe.py", title="MoMoe"),
        st.Page("TLDR.py", title="Summa.ry"),
        st.Page("Ui.py", title="RLHF"),
        st.Page("Andrei.py", title="AndreiSFTPPO")
    ]
}

pg = st.navigation(pages)
pg.run()



