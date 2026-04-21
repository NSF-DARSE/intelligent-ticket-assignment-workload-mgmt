from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin

import requests
from dotenv import load_dotenv


DEFAULT_ZONE_INFO_URL = "https://webservices.autotask.net/ATServicesRest/V1.0/zoneInformation"


@dataclass
class AutotaskConfig:
    base_url: str
    username: str
    secret: str
    integration_code: str
    page_size: int = 500

    @classmethod
    def from_env(cls, env_path: str | None = None) -> "AutotaskConfig":
        load_dotenv(env_path)

        username = os.getenv("AUTOTASK_API_USERNAME", "").strip()
        secret = os.getenv("AUTOTASK_API_SECRET", "").strip()
        integration_code = os.getenv("AUTOTASK_API_INTEGRATION_CODE", "").strip()
        base_url = os.getenv("AUTOTASK_API_BASE_URL", "").strip()
        page_size = int(os.getenv("AUTOTASK_PAGE_SIZE", "500"))

        if not base_url:
            base_url = discover_base_url(username=username)

        missing = [
            name
            for name, value in {
                "AUTOTASK_API_BASE_URL": base_url,
                "AUTOTASK_API_USERNAME": username,
                "AUTOTASK_API_SECRET": secret,
                "AUTOTASK_API_INTEGRATION_CODE": integration_code,
            }.items()
            if not value
        ]
        if missing:
            missing_text = ", ".join(missing)
            raise ValueError(
                "Missing Autotask sandbox configuration. "
                f"Set these environment variables in .env: {missing_text}"
            )

        return cls(
            base_url=ensure_trailing_slash(base_url),
            username=username,
            secret=secret,
            integration_code=integration_code,
            page_size=page_size,
        )


def ensure_trailing_slash(value: str) -> str:
    return value if value.endswith("/") else f"{value}/"


def discover_base_url(username: str) -> str:
    zone_info_url = os.getenv("AUTOTASK_ZONE_INFO_URL", DEFAULT_ZONE_INFO_URL).strip()
    if not username or not zone_info_url:
        return ""

    response = requests.get(zone_info_url, params={"user": username}, timeout=30)
    response.raise_for_status()

    payload = response.json()
    zone_url = payload.get("url") or payload.get("baseUrl") or payload.get("zoneUrl") or ""
    return ensure_trailing_slash(zone_url)


class AutotaskApiClient:
    def __init__(self, config: AutotaskConfig) -> None:
        self.config = config
        self.session = requests.Session()
        # Ignore machine-level proxy settings so sandbox calls go directly to Autotask.
        self.session.trust_env = False
        self.session.headers.update(
            {
                "ApiIntegrationCode": config.integration_code,
                "UserName": config.username,
                "Secret": config.secret,
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
        )

    def _build_url(self, path: str) -> str:
        return urljoin(self.config.base_url, path.lstrip("/"))

    def _get_json(self, url: str) -> dict[str, Any]:
        response = self.session.get(url, timeout=60)
        response.raise_for_status()
        return response.json()

    def _post_json_url(self, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        response = self.session.post(url, json=payload or {}, timeout=60)
        response.raise_for_status()
        return response.json()

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = self.session.post(self._build_url(path), json=payload, timeout=60)
        response.raise_for_status()
        return response.json()

    def query_entity(
        self,
        entity_name: str,
        filters: list[dict[str, Any]] | None = None,
        max_records: int | None = None,
    ) -> list[dict[str, Any]]:
        body: dict[str, Any] = {"filter": filters or [{"field": "id", "op": "gte", "value": 0}]}
        payload = self._post_json(f"{entity_name}/query", body)

        items = list(payload.get("items", []))
        next_page_url = ((payload.get("pageDetails") or {}).get("nextPageUrl")) or payload.get("nextPageUrl")

        while next_page_url:
            # Autotask query pagination uses a POST-only "query/next" endpoint.
            # This sandbox also expects the original query body to be posted again.
            page_payload = self._post_json_url(next_page_url, body)
            items.extend(page_payload.get("items", []))
            next_page_url = ((page_payload.get("pageDetails") or {}).get("nextPageUrl")) or page_payload.get(
                "nextPageUrl"
            )

            if max_records and len(items) >= max_records:
                return items[:max_records]

        return items[:max_records] if max_records else items

    def fetch_reference_maps(self) -> dict[str, dict[Any, str]]:
        reference_maps: dict[str, dict[Any, str]] = {}

        entity_specs = {
            "companies": ("Companies", ["companyName", "name"]),
            "resources": ("Resources", ["fullName", "userName", "firstName"]),
            "queues": ("Queues", ["name"]),
            "contracts": ("Contracts", ["contractName", "name"]),
            "service_levels": ("ServiceLevelAgreements", ["name"]),
        }

        for map_name, (entity_name, name_fields) in entity_specs.items():
            try:
                records = self.query_entity(entity_name, max_records=2000)
            except requests.RequestException:
                reference_maps[map_name] = {}
                continue

            lookup: dict[Any, str] = {}
            for record in records:
                record_id = record.get("id")
                if record_id is None:
                    continue
                for field_name in name_fields:
                    name = record.get(field_name)
                    if name not in (None, ""):
                        lookup[record_id] = str(name)
                        break
            reference_maps[map_name] = lookup

        return reference_maps

    def fetch_picklist_maps(self, entity_name: str) -> dict[str, dict[str, str]]:
        payload = self._get_json(self._build_url(f"{entity_name}/entityInformation/fields"))
        picklist_maps: dict[str, dict[str, str]] = {}

        for field in payload.get("fields", []):
            field_name = field.get("name")
            if not field_name or not field.get("isPickList"):
                continue

            values = field.get("picklistValues") or []
            if not values:
                continue

            value_map = {
                str(item.get("value")): str(item.get("label"))
                for item in values
                if item.get("value") not in (None, "") and item.get("label") not in (None, "")
            }
            if value_map:
                picklist_maps[field_name] = value_map

        return picklist_maps
