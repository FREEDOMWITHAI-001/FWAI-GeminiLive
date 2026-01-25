"""
Meta WhatsApp Token Manager
Auto-refreshes access tokens before they expire
"""

import asyncio
import httpx
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger
from src.core.config import config

# Token storage file
TOKEN_FILE = Path(__file__).parent.parent.parent / ".meta_token"


class MetaTokenManager:
    """Manages Meta access tokens with auto-refresh"""

    def __init__(self):
        self._access_token = config.meta_access_token
        self._token_expiry = None
        self._refresh_task = None
        self._initialized = False

    @property
    def access_token(self) -> str:
        """Get the current access token"""
        return self._access_token

    async def initialize(self):
        """Initialize token manager - exchange for long-lived token if possible"""
        if self._initialized:
            return

        # Try to load saved token
        if self._load_saved_token():
            logger.info("Loaded saved Meta access token")
        elif config.meta_app_id and config.meta_app_secret:
            # Exchange short-lived token for long-lived token
            await self._exchange_for_long_lived_token()
        else:
            logger.warning("META_APP_ID and META_APP_SECRET not set - using provided token without refresh")

        self._initialized = True

        # Start background refresh task
        if self._token_expiry and config.meta_app_id and config.meta_app_secret:
            self._refresh_task = asyncio.create_task(self._auto_refresh_loop())

    def _load_saved_token(self) -> bool:
        """Load token from file if it exists and is valid"""
        try:
            if TOKEN_FILE.exists():
                content = TOKEN_FILE.read_text().strip()
                lines = content.split("\n")
                if len(lines) >= 2:
                    self._access_token = lines[0]
                    expiry_str = lines[1]
                    self._token_expiry = datetime.fromisoformat(expiry_str)

                    # Check if token is still valid (with 1 hour buffer)
                    if self._token_expiry > datetime.now() + timedelta(hours=1):
                        logger.info(f"Token valid until {self._token_expiry}")
                        return True
                    else:
                        logger.info("Saved token expired or expiring soon")
        except Exception as e:
            logger.error(f"Error loading saved token: {e}")
        return False

    def _save_token(self):
        """Save token to file"""
        try:
            if self._token_expiry:
                content = f"{self._access_token}\n{self._token_expiry.isoformat()}"
                TOKEN_FILE.write_text(content)
                logger.info(f"Saved token to {TOKEN_FILE}")
        except Exception as e:
            logger.error(f"Error saving token: {e}")

    async def _exchange_for_long_lived_token(self):
        """Exchange short-lived token for long-lived token (60 days)"""
        try:
            logger.info("Exchanging for long-lived token...")

            url = "https://graph.facebook.com/v22.0/oauth/access_token"
            params = {
                "grant_type": "fb_exchange_token",
                "client_id": config.meta_app_id,
                "client_secret": config.meta_app_secret,
                "fb_exchange_token": self._access_token
            }

            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params)

                if response.status_code == 200:
                    data = response.json()
                    self._access_token = data.get("access_token")
                    expires_in = data.get("expires_in", 5184000)  # Default 60 days

                    self._token_expiry = datetime.now() + timedelta(seconds=expires_in)
                    self._save_token()

                    logger.info(f"Got long-lived token, expires: {self._token_expiry}")
                else:
                    logger.error(f"Token exchange failed: {response.status_code} - {response.text}")

        except Exception as e:
            logger.error(f"Error exchanging token: {e}")

    async def _refresh_token(self):
        """Refresh the long-lived token"""
        try:
            logger.info("Refreshing Meta access token...")

            # For long-lived tokens, we can refresh them before they expire
            url = "https://graph.facebook.com/v22.0/oauth/access_token"
            params = {
                "grant_type": "fb_exchange_token",
                "client_id": config.meta_app_id,
                "client_secret": config.meta_app_secret,
                "fb_exchange_token": self._access_token
            }

            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params)

                if response.status_code == 200:
                    data = response.json()
                    self._access_token = data.get("access_token")
                    expires_in = data.get("expires_in", 5184000)

                    self._token_expiry = datetime.now() + timedelta(seconds=expires_in)
                    self._save_token()

                    logger.info(f"Token refreshed, new expiry: {self._token_expiry}")
                    return True
                else:
                    logger.error(f"Token refresh failed: {response.status_code} - {response.text}")
                    return False

        except Exception as e:
            logger.error(f"Error refreshing token: {e}")
            return False

    async def _auto_refresh_loop(self):
        """Background task to auto-refresh token before expiry"""
        while True:
            try:
                if self._token_expiry:
                    # Refresh 24 hours before expiry
                    time_until_expiry = (self._token_expiry - datetime.now()).total_seconds()
                    refresh_in = max(time_until_expiry - 86400, 3600)  # At least 1 hour from now

                    logger.info(f"Token refresh scheduled in {refresh_in/3600:.1f} hours")
                    await asyncio.sleep(refresh_in)

                    if await self._refresh_token():
                        continue
                    else:
                        # Retry in 1 hour if refresh failed
                        await asyncio.sleep(3600)
                else:
                    # No expiry set, check every 24 hours
                    await asyncio.sleep(86400)

            except asyncio.CancelledError:
                logger.info("Token refresh task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in auto-refresh loop: {e}")
                await asyncio.sleep(3600)

    async def debug_token(self) -> dict:
        """Debug the current token to check its validity"""
        try:
            url = "https://graph.facebook.com/debug_token"
            params = {
                "input_token": self._access_token,
                "access_token": f"{config.meta_app_id}|{config.meta_app_secret}"
            }

            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params)
                return response.json()
        except Exception as e:
            return {"error": str(e)}


# Global token manager instance
token_manager = MetaTokenManager()


async def get_access_token() -> str:
    """Get the current valid access token"""
    if not token_manager._initialized:
        await token_manager.initialize()
    return token_manager.access_token
