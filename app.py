"""
app.py
Streamlit entry point (thin). Keeps UI orchestration minimal and delegates logic to modules in ui/ and models/.
"""
import streamlit as st

# Configure the page layout to wide for the whole app
st.set_page_config(layout='wide', page_title='Options Pricing – Professional', initial_sidebar_state='expanded')

# The legacy dashboard.py remains the canonical UI for now; import it to preserve behavior.
# Future work: migrate dashboard logic into ui.layout and call a main() handler here.

try:
    from ui.layout import render_dashboard
    render_dashboard()
except Exception as e:
    st.write('Unable to render dashboard:', e)


if __name__ == '__main__':
    # Streamlit runs the file top-level; nothing to do here for now.
    pass
