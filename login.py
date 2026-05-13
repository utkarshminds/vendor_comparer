import streamlit as st

def check_credentials(username: str, password: str) -> bool:
    """Check if the provided credentials are correct."""
    valid_username = st.secrets.get("username", "test")
    valid_password = st.secrets.get("password", "0123456789p")
    return username == valid_username and password == valid_password

def main():
    st.set_page_config(page_title="Login", layout="centered")
    
    # Initialize session state for login
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    
    if st.session_state.logged_in:
        st.success("✓ Successfully logged in!")
        st.write(f"Welcome, {st.session_state.username}!")
        
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.rerun()
    else:
        st.title("Login")
        
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit_button = st.form_submit_button("Login")
        
        if submit_button:
            if check_credentials(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid username or password. Please try again.")

if __name__ == "__main__":
    main()
