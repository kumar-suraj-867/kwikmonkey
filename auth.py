"""Fyers OAuth2 authentication flow."""

import os
import sys
import webbrowser

from fyers_apiv3 import fyersModel

import config


def generate_auth_url() -> str:
    """Generate the Fyers login URL for OAuth2 authorization."""
    session = fyersModel.SessionModel(
        client_id=config.FYERS_APP_ID,
        secret_key=config.FYERS_SECRET_KEY,
        redirect_uri=config.FYERS_REDIRECT_URI,
        response_type="code",
        grant_type="authorization_code",
    )
    return session.generate_authcode()


def generate_token(auth_code: str) -> str:
    """Exchange authorization code for access token."""
    session = fyersModel.SessionModel(
        client_id=config.FYERS_APP_ID,
        secret_key=config.FYERS_SECRET_KEY,
        redirect_uri=config.FYERS_REDIRECT_URI,
        response_type="code",
        grant_type="authorization_code",
    )
    session.set_token(auth_code)
    response = session.generate_token()

    if response.get("s") != "ok" and "access_token" not in response:
        raise RuntimeError(f"Token generation failed: {response}")

    access_token = response["access_token"]
    save_token(access_token)
    return access_token


def save_token(token: str) -> None:
    """Persist access token to file."""
    with open(config.TOKEN_FILE, "w") as f:
        f.write(token)


def load_token() -> str | None:
    """Load access token from file, if it exists."""
    if os.path.exists(config.TOKEN_FILE):
        with open(config.TOKEN_FILE) as f:
            token = f.read().strip()
        return token if token else None
    return None


def validate_token(token: str) -> bool:
    """Check if the token is still valid by making a profile request."""
    fyers = fyersModel.FyersModel(
        client_id=config.FYERS_APP_ID,
        token=token,
        is_async=False,
        log_path="",
    )
    try:
        response = fyers.get_profile()
        return response.get("s") == "ok"
    except Exception:
        return False


def get_valid_token() -> str | None:
    """Return a valid token if available, else None."""
    token = load_token()
    if token and validate_token(token):
        return token
    return None


def run_auth_flow() -> str:
    """Interactive CLI auth flow: opens browser, takes auth code, returns token."""
    if not config.FYERS_APP_ID or not config.FYERS_SECRET_KEY:
        print("ERROR: Set FYERS_APP_ID and FYERS_SECRET_KEY in your .env file.")
        print("Get these from https://myapi.fyers.in/dashboard")
        sys.exit(1)

    auth_url = generate_auth_url()
    print(f"\nOpening browser for Fyers login...\n")
    print(f"If browser doesn't open, visit:\n{auth_url}\n")
    webbrowser.open(auth_url)

    print("After logging in, you'll be redirected to your redirect URI.")
    print("Copy the 'auth_code' parameter from the URL.\n")
    auth_code = input("Paste auth_code here: ").strip()

    if not auth_code:
        print("No auth code provided. Exiting.")
        sys.exit(1)

    token = generate_token(auth_code)
    print("\nAuthentication successful! Token saved.\n")
    return token


if __name__ == "__main__":
    run_auth_flow()
