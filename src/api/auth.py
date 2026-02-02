"""
Authentication and User Resolution
Flexible auth system supporting JWT, cookies, OAuth tokens
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..core.base import UserContext

logger = logging.getLogger(__name__)


class UserResolver(ABC):
    """
    Abstract user resolver - extracts user identity from requests

    Implement this for your authentication system:
    - JWT tokens
    - Session cookies
    - OAuth tokens
    - API keys
    """

    @abstractmethod
    async def resolve(self, request: Any) -> UserContext:
        """
        Resolve user from request

        Args:
            request: FastAPI Request object

        Returns:
            UserContext with user identity and permissions
        """
        pass

    @abstractmethod
    async def get_permissions(self, user_id: str) -> Dict[str, Any]:
        """
        Get permissions for a user

        Returns:
            Dict containing:
            - roles: List of role names
            - allowed_schemas: Schemas user can access
            - allowed_tables: Tables user can access
            - sql_filters: Row-level security filters
        """
        pass


class JWTUserResolver(UserResolver):
    """
    JWT-based user resolver

    Extracts user from Authorization header Bearer token
    """

    def __init__(
        self,
        secret_key: str,
        algorithm: str = "HS256",
        permissions_loader: Optional[callable] = None,
    ):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.permissions_loader = permissions_loader
        self._jwt = None

    def _get_jwt(self):
        if self._jwt is None:
            try:
                import jwt
                self._jwt = jwt
            except ImportError:
                raise ImportError("PyJWT required: pip install PyJWT")
        return self._jwt

    async def resolve(self, request: Any) -> UserContext:
        """Resolve user from JWT token"""
        jwt_lib = self._get_jwt()

        # Get token from header
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return self._anonymous_user()

        token = auth_header[7:]

        try:
            payload = jwt_lib.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
            )

            user_id = payload.get("sub") or payload.get("user_id")
            if not user_id:
                return self._anonymous_user()

            permissions = await self.get_permissions(user_id)

            return UserContext(
                user_id=user_id,
                session_id=payload.get("session_id", ""),
                roles=permissions.get("roles", []),
                permissions=permissions,
                metadata={
                    "email": payload.get("email"),
                    "name": payload.get("name"),
                    "exp": payload.get("exp"),
                },
            )

        except jwt_lib.ExpiredSignatureError:
            logger.warning("Expired JWT token")
            return self._anonymous_user()
        except jwt_lib.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {e}")
            return self._anonymous_user()

    async def get_permissions(self, user_id: str) -> Dict[str, Any]:
        """Get user permissions from loader or default"""
        if self.permissions_loader:
            return await self.permissions_loader(user_id)

        # Default permissions
        return {
            "roles": ["user"],
            "allowed_schemas": [],
            "allowed_tables": [],
            "sql_filters": {},
        }

    def _anonymous_user(self) -> UserContext:
        """Return anonymous user context"""
        return UserContext(
            user_id="anonymous",
            roles=["anonymous"],
            permissions={"roles": ["anonymous"]},
        )


class CookieUserResolver(UserResolver):
    """
    Cookie-based session resolver

    Works with traditional session cookies
    """

    def __init__(
        self,
        session_cookie_name: str = "session",
        session_store: Optional[Any] = None,
    ):
        self.cookie_name = session_cookie_name
        self.session_store = session_store

    async def resolve(self, request: Any) -> UserContext:
        """Resolve user from session cookie"""
        session_id = request.cookies.get(self.cookie_name)

        if not session_id or not self.session_store:
            return self._anonymous_user()

        session = await self.session_store.get(session_id)
        if not session:
            return self._anonymous_user()

        user_id = session.get("user_id")
        if not user_id:
            return self._anonymous_user()

        permissions = await self.get_permissions(user_id)

        return UserContext(
            user_id=user_id,
            session_id=session_id,
            roles=permissions.get("roles", []),
            permissions=permissions,
            metadata=session.get("metadata", {}),
        )

    async def get_permissions(self, user_id: str) -> Dict[str, Any]:
        """Get permissions - implement based on your system"""
        return {
            "roles": ["user"],
            "allowed_schemas": [],
            "allowed_tables": [],
            "sql_filters": {},
        }

    def _anonymous_user(self) -> UserContext:
        return UserContext(
            user_id="anonymous",
            roles=["anonymous"],
            permissions={"roles": ["anonymous"]},
        )


class APIKeyUserResolver(UserResolver):
    """
    API key-based resolver

    For service-to-service or programmatic access
    """

    def __init__(
        self,
        api_keys: Dict[str, Dict[str, Any]],
        header_name: str = "X-API-Key",
    ):
        """
        Args:
            api_keys: Dict mapping API keys to user info
                {"key123": {"user_id": "service1", "roles": ["admin"]}}
            header_name: Header name for API key
        """
        self.api_keys = api_keys
        self.header_name = header_name

    async def resolve(self, request: Any) -> UserContext:
        """Resolve user from API key"""
        api_key = request.headers.get(self.header_name)

        if not api_key or api_key not in self.api_keys:
            return self._anonymous_user()

        key_info = self.api_keys[api_key]

        return UserContext(
            user_id=key_info.get("user_id", "api_user"),
            roles=key_info.get("roles", ["api"]),
            permissions=key_info.get("permissions", {}),
            metadata={"api_key": api_key[:8] + "..."},
        )

    async def get_permissions(self, user_id: str) -> Dict[str, Any]:
        """Get permissions from api_keys config"""
        for key_info in self.api_keys.values():
            if key_info.get("user_id") == user_id:
                return key_info.get("permissions", {})
        return {}

    def _anonymous_user(self) -> UserContext:
        return UserContext(
            user_id="anonymous",
            roles=["anonymous"],
            permissions={"roles": ["anonymous"]},
        )


class CompositeUserResolver(UserResolver):
    """
    Composite resolver that tries multiple resolvers in order
    """

    def __init__(self, resolvers: List[UserResolver]):
        self.resolvers = resolvers

    async def resolve(self, request: Any) -> UserContext:
        """Try each resolver until one succeeds"""
        for resolver in self.resolvers:
            try:
                user = await resolver.resolve(request)
                if user.user_id != "anonymous":
                    return user
            except Exception as e:
                logger.debug(f"Resolver {resolver.__class__.__name__} failed: {e}")
                continue

        # All resolvers failed, return anonymous
        return UserContext(
            user_id="anonymous",
            roles=["anonymous"],
            permissions={"roles": ["anonymous"]},
        )

    async def get_permissions(self, user_id: str) -> Dict[str, Any]:
        """Try to get permissions from any resolver"""
        for resolver in self.resolvers:
            try:
                perms = await resolver.get_permissions(user_id)
                if perms:
                    return perms
            except Exception:
                continue
        return {}
