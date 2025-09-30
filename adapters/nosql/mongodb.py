"""MongoDB query adapter bridging to legacy execution logic."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Dict, Optional, Union

from ...core.connections import BaseConnectionConfig
from ...core.connections.mongodb import MongoDBConfig
from ...core.query_composer import BaseQueryComposer
from ...utils.helpers import get_logger


class MongoDBAdapter(BaseQueryComposer):
    """MongoDB-specific query adapter that delegates to the legacy async client."""

    _legacy_client_cls = None

    def __init__(self, config: Optional[BaseConnectionConfig] = None):
        if config is None:
            # Provide a sensible default configuration for local development
            config = MongoDBConfig(database_name="default")
        super().__init__(config)
        self.logger = get_logger("text2query.adapters.mongodb")
        self._legacy_client = None

    async def sql_execution(self, command: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Execute MongoDB command using the legacy async client implementation."""

        client = await self._ensure_legacy_client()
        return await client.sql_execution(command)

    async def close_conn(self) -> None:
        """Close the underlying legacy client connection."""

        if self._legacy_client is not None:
            try:
                await self._legacy_client.close_conn()
            finally:
                self._legacy_client = None

    async def _ensure_legacy_client(self):
        """Lazily instantiate and prepare the legacy Mongo client."""

        client_cls = self._load_legacy_client_class()
        connection_info = self._build_connection_info()

        if self._legacy_client is None:
            self._legacy_client = client_cls(connection_info)
        else:
            # Keep connection information up-to-date with latest configuration
            self._legacy_client.connection_info = connection_info

        conn_result = await self._legacy_client.get_conn()
        if isinstance(conn_result, str):
            raise RuntimeError(f"MongoDB connection failed: {conn_result}")
        return self._legacy_client

    def _build_connection_info(self) -> Dict[str, Any]:
        if not isinstance(self.config, BaseConnectionConfig):
            raise RuntimeError("MongoDBAdapter requires a valid MongoDB configuration")

        if isinstance(self.config, MongoDBConfig):
            cfg = self.config
        else:
            cfg = MongoDBConfig(
                database_name=self.config.database_name,
                timeout=self.config.timeout,
                extra_params=self.config.extra_params,
            )
            for attr in ["host", "port", "username", "password", "auth_database", "replica_set", "ssl", "ssl_cert_reqs", "read_preference", "max_pool_size"]:
                if hasattr(self.config, attr):
                    setattr(cfg, attr, getattr(self.config, attr))

        connection_info = {
            "connection_string": cfg.to_connection_string(),
            "database": cfg.database_name,
            "collections": cfg.extra_params.get("collections", []),
        }
        return connection_info

    @classmethod
    def _load_legacy_client_class(cls):
        """Load the legacy MongoClient class from old_sample module."""

        if cls._legacy_client_cls is not None:
            return cls._legacy_client_cls

        legacy_path = Path(__file__).resolve().parents[2] / "old_sample" / "mongo.py"
        spec = importlib.util.spec_from_file_location("text2query_legacy.mongo", legacy_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load legacy Mongo client from {legacy_path}")

        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except ImportError as exc:
            raise RuntimeError(
                "The legacy MongoDB client requires additional dependencies (motor, bson)."
            ) from exc

        if not hasattr(module, "MongoClient"):
            raise RuntimeError("Legacy MongoClient class not found in old_sample.mongo")

        cls._legacy_client_cls = module.MongoClient
        return cls._legacy_client_cls
