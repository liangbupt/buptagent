import json
import os
import sqlite3
from typing import Any, Dict, List

from app.core.config import settings


class ToolDataProvider:
    def __init__(self) -> None:
        self._mode = (settings.TOOL_DATA_MODE or "json").strip().lower()
        self._json_path = settings.TOOL_DATA_JSON_PATH
        self._sqlite_path = settings.TOOL_DATA_SQLITE_PATH
        self._json_data: Dict[str, Any] = {}
        self._json_mtime = 0.0

    def _ensure_json_loaded(self) -> None:
        if self._mode != "json":
            return
        path = self._json_path
        if not os.path.exists(path):
            self._json_data = {}
            self._json_mtime = 0.0
            return
        mtime = os.path.getmtime(path)
        if mtime <= self._json_mtime and self._json_data:
            return
        with open(path, "r", encoding="utf-8") as f:
            self._json_data = json.load(f)
        self._json_mtime = mtime

    def _json_get(self, key: str) -> List[Dict[str, Any]]:
        self._ensure_json_loaded()
        val = self._json_data.get(key, [])
        return val if isinstance(val, list) else []

    def get_free_classrooms(self, building: str, time_slot: str) -> str:
        if self._mode == "json":
            rows = self._json_get("free_classrooms")
            for row in rows:
                if row.get("building") == building and row.get("time_slot") == time_slot:
                    rooms = ", ".join(row.get("classrooms", []))
                    return f"DataSource=json: In {building} at {time_slot}, classrooms {rooms} are free."
        return f"DataSource=mock: In {building} at {time_slot}, classrooms 101, 102 are free."

    def get_audit_course_suggestion(self, topic: str, preferred_time: str = "afternoon") -> str:
        normalized = topic.lower().strip()
        if self._mode == "json":
            rows = self._json_get("audit_courses")
            for row in rows:
                row_topic = str(row.get("topic", "")).lower()
                if row_topic and row_topic in normalized:
                    return (
                        "DataSource=json: "
                        f"For topic '{topic}', suggested course is '{row.get('course')}' around {row.get('preferred_time', preferred_time)}."
                    )
        return (
            "DataSource=mock: "
            f"For topic '{topic}', suggested audit course is 'AI Fundamentals' with open seats around {preferred_time}."
        )

    def get_dining_recommendation(self, preference: str) -> str:
        pref = preference.lower()
        if self._mode == "json":
            rows = self._json_get("dining")
            if "spicy" in pref or "辣" in preference:
                for row in rows:
                    if str(row.get("tag", "")).lower() == "spicy":
                        return f"DataSource=json: {row.get('text')}"
            for row in rows:
                if str(row.get("tag", "")).lower() == "light":
                    return f"DataSource=json: {row.get('text')}"
        if "spicy" in pref or "辣" in preference:
            return "DataSource=mock: Recommend the spicy hotpot on the 2nd floor of Student Dining Hall."
        return "DataSource=mock: Recommend the light set meal on the 1st floor of Xinyuan Dining Hall."

    def get_flea_market_items(self, keyword: str, budget: str = "不限") -> str:
        if self._mode == "json":
            rows = self._json_get("flea_market")
            matched = [row.get("text", "") for row in rows if keyword.lower() in str(row.get("keyword", "")).lower()]
            if matched:
                return f"DataSource=json: Found flea market items: {', '.join(matched)}"
        return (
            "DataSource=mock: "
            f"Found flea market items for '{keyword}' under budget '{budget}': used monitor 300 CNY, desk lamp 45 CNY."
        )

    def get_campus_feedback(self, topic: str) -> str:
        lower_topic = topic.lower()
        if self._mode == "json":
            rows = self._json_get("feedback")
            for row in rows:
                if str(row.get("topic", "")).lower() in lower_topic:
                    return f"DataSource=json: {row.get('text')}"
        return f"DataSource=mock: Recent campus feedback on '{topic}' is mostly positive with suggestions on queue time."


provider = ToolDataProvider()
